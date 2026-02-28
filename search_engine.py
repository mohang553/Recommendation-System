"""
search_engine.py
Elasticsearch-powered product search.
Handles indexing products from MySQL and searching with full-text + filters.
"""

import os
import logging
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

load_dotenv()

logger    = logging.getLogger(__name__)
ES_HOST   = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX  = os.getenv("ES_INDEX", "amazon_products")


# ── Connect ───────────────────────────────────────────────────────────────────

def get_es_client() -> Elasticsearch:
    """Create and return Elasticsearch client."""
    client = Elasticsearch(ES_HOST)
    if not client.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {ES_HOST}")
    logger.info("✓ Connected to Elasticsearch")
    return client


# ── Index Definition ──────────────────────────────────────────────────────────

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "product_id"   : {"type": "keyword"},        # exact match only
            "product_name" : {"type": "text",            # full-text search
                              "analyzer": "english",     # handles stemming (shoe→shoes)
                              "fields": {
                                  "keyword": {"type": "keyword"}  # for sorting
                              }},
            "category_id"  : {"type": "integer"},
            "avg_rating"   : {"type": "float"},
            "total_reviews": {"type": "integer"},
            "price"        : {"type": "float"},
        }
    },
    "settings": {
        "number_of_shards"  : 1,
        "number_of_replicas": 0    # single node setup
    }
}


# ── Step 1: Create Index ──────────────────────────────────────────────────────

def create_index(es: Elasticsearch):
    """Create the products index with mapping. Skip if already exists."""
    if es.indices.exists(index=ES_INDEX):
        logger.info(f"Index '{ES_INDEX}' already exists — skipping creation")
        return

    es.indices.create(index=ES_INDEX, body=INDEX_MAPPING)
    logger.info(f"✓ Index '{ES_INDEX}' created")


# ── Step 2: Load Products from MySQL ─────────────────────────────────────────

def load_products_from_db(engine) -> pd.DataFrame:
    """Pull all products from MySQL for indexing."""
    logger.info("Loading products from MySQL for indexing ...")
    df = pd.read_sql("""
        SELECT
            asin    AS product_id,
            title   AS product_name,
            stars   AS avg_rating,
            reviews AS total_reviews,
            price,
            category_id
        FROM amazon_products
        WHERE title IS NOT NULL
    """, engine)
    logger.info(f" → {len(df):,} products loaded")
    return df


# ── Step 3: Index Products into Elasticsearch ────────────────────────────────

def index_products(es: Elasticsearch, df: pd.DataFrame):
    """
    Bulk index all products into Elasticsearch.
    Uses helpers.bulk() for efficiency — much faster than indexing one by one.
    """
    logger.info(f"Indexing {len(df):,} products into Elasticsearch ...")

    def generate_docs():
        for _, row in df.iterrows():
            yield {
                "_index": ES_INDEX,
                "_id"   : row["product_id"],   # use product_id as ES doc ID
                "_source": {
                    "product_id"   : row["product_id"],
                    "product_name" : str(row["product_name"]) if row["product_name"] else "",
                    "avg_rating"   : float(row["avg_rating"])    if pd.notna(row["avg_rating"])    else 0.0,
                    "total_reviews": int(row["total_reviews"])   if pd.notna(row["total_reviews"]) else 0,
                    "price"        : float(row["price"])         if pd.notna(row["price"])         else 0.0,
                    "category_id"  : int(row["category_id"])     if pd.notna(row["category_id"])   else 0,
                }
            }

    success, failed = helpers.bulk(es, generate_docs(), raise_on_error=False)
    logger.info(f" → Indexed: {success:,} | Failed: {len(failed):,}")


# ── Step 4: Search ────────────────────────────────────────────────────────────

def search_products(
    es          : Elasticsearch,
    query       : str,
    top_n       : int   = 10,
    min_rating  : float = 0.0,
    category_id : int   = None,
    min_price   : float = None,
    max_price   : float = None,
) -> list:
    """
    Full-text search with optional filters.
    
    Scoring:
      - Full-text match on product_name (boosted)
      - Filtered by rating, category, price range
      - Results re-ranked by: text score + avg_rating boost
    """

    # ── Filters (hard constraints) ────────────────────────────────────────────
    filters = [
        {"range": {"avg_rating": {"gte": min_rating}}}
    ]

    if category_id:
        filters.append({"term": {"category_id": category_id}})

    if min_price is not None or max_price is not None:
        price_range = {}
        if min_price is not None:
            price_range["gte"] = min_price
        if max_price is not None:
            price_range["lte"] = max_price
        filters.append({"range": {"price": price_range}})

    # ── Full ES Query ─────────────────────────────────────────────────────────
    es_query = {
        "size": top_n,
        "query": {
            "function_score": {
                # Step 1: Full text match
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query"    : query,
                                    "fields"   : ["product_name^3"],  # ^3 = boost title matches
                                    "fuzziness": "AUTO",              # handles typos
                                    "operator" : "or"
                                }
                            }
                        ],
                        "filter": filters
                    }
                },
                # Step 2: Boost by avg_rating so higher rated products rank up
                "functions": [
                    {
                        "field_value_factor": {
                            "field"   : "avg_rating",
                            "factor"  : 1.5,          # weight of rating boost
                            "modifier": "log1p",       # smooth out extreme values
                            "missing" : 1              # default if field missing
                        }
                    }
                ],
                "boost_mode": "multiply"   # final_score = text_score × rating_boost
            }
        },
        # ── Sort: relevance score first, then rating as tiebreaker ────────────
        "sort": [
            {"_score"     : {"order": "desc"}},
            {"avg_rating" : {"order": "desc"}},
            {"total_reviews": {"order": "desc"}}
        ]
    }

    response = es.search(index=ES_INDEX, body=es_query)
    hits     = response["hits"]["hits"]

    return [
        {
            **hit["_source"],
            "relevance_score": round(hit["_score"], 4)
        }
        for hit in hits
    ]


# ── Load Search Model (called at FastAPI startup) ─────────────────────────────

def load_search_engine(engine) -> Elasticsearch:
    """
    Connect to ES, create index if needed, index products if empty.
    Call this once at FastAPI startup.
    """
    es = get_es_client()
    create_index(es)

    # Only index if the index is empty (avoids re-indexing on every restart)
    count = es.count(index=ES_INDEX)["count"]
    if count == 0:
        logger.info("Index is empty — loading products from MySQL ...")
        df = load_products_from_db(engine)
        index_products(es, df)
    else:
        logger.info(f"✓ Index already has {count:,} products — skipping indexing")

    return es