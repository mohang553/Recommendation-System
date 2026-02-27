"""

Content-Based Filtering using TF-IDF on product text features.

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


# ── Step 1: Load Product Catalog ─────────────────────────────────────────────

def load_product_catalog(engine) -> pd.DataFrame:
    logger.info("Loading product catalog ...")
    df = pd.read_sql("""
        SELECT 
            asin        AS product_id,
            title,
            category_id,
            stars       AS avg_rating,
            reviews     AS total_reviews,
            price
        FROM amazon_products
        WHERE title IS NOT NULL
    """, engine)
    logger.info(f" → {len(df):,} products loaded")
    return df


# ── Step 2: Build TF-IDF Matrix ──────────────────────────────────────────────

def build_tfidf_matrix(product_df: pd.DataFrame):
    logger.info("Building TF-IDF matrix ...")

    df = product_df.copy()
    df["category_id"] = df["category_id"].fillna("").astype(str)
    df["title"]       = df["title"].fillna("")

    df["corpus"] = df["title"] + " " + df["category_id"] + " " + df["category_id"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        sublinear_tf=True
    )

    tfidf_matrix  = vectorizer.fit_transform(df["corpus"])
    product_index = pd.Series(range(len(df)), index=df["product_id"])

    logger.info(f" → {tfidf_matrix.shape[0]:,} products × {tfidf_matrix.shape[1]:,} features")
    return tfidf_matrix, product_index, vectorizer


# ── Step 3: Get Content Recommendations ──────────────────────────────────────

def get_content_recommendations(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    tfidf_matrix,
    product_index: pd.Series,
    top_n: int = 10,
    top_rated_n: int = 5
) -> pd.DataFrame:
    if user_id not in user_item_matrix.index:
        logger.warning(f"User {user_id} not found in matrix")
        return pd.DataFrame()

    user_ratings = user_item_matrix.loc[user_id]
    rated        = user_ratings[user_ratings > 0]

    if rated.empty:
        logger.warning(f"User {user_id} has no ratings")
        return pd.DataFrame()

    seeds = [p for p in rated.nlargest(top_rated_n).index if p in product_index.index]
    if not seeds:
        return pd.DataFrame()

    seed_idx     = [product_index[p] for p in seeds]
    user_profile = np.asarray(tfidf_matrix[seed_idx].mean(axis=0))
    scores       = cosine_similarity(user_profile, tfidf_matrix).flatten()
    already_seen = set(rated.index)

    results = [
        {"product_id": pid, "content_score": round(float(scores[idx]), 4)}
        for pid, idx in product_index.items()
        if pid not in already_seen
    ]

    rec_df         = (pd.DataFrame(results)
                        .sort_values("content_score", ascending=False)
                        .head(top_n)
                        .reset_index(drop=True))
    rec_df["rank"] = range(1, len(rec_df) + 1)
    return rec_df


# ── Step 4: Load Content Model ────────────────────────────────────────────────

def load_content_model(engine):
    product_df                               = load_product_catalog(engine)
    tfidf_matrix, product_index, vectorizer  = build_tfidf_matrix(product_df)
    logger.info("✓ Content model ready")
    return product_df, tfidf_matrix, product_index, vectorizer


# ── Main (standalone) ─────────────────────────────────────────────────────────

def main():
    from collaborative import build_engine, load_model, enrich_with_product_details

    engine = build_engine()
    matrix, _ = load_model(engine)

    product_df, tfidf_matrix, product_index, _ = load_content_model(engine)

    sample_user_id = matrix.index[0]
    logger.info(f"\n── Content Recs for User {sample_user_id} ──")

    rec_df = get_content_recommendations(sample_user_id, matrix, tfidf_matrix, product_index)
    rec_df = enrich_with_product_details(rec_df, engine)

    print("\n" + "="*70)
    print(f" Top 10 Content-Based Recs for User: {sample_user_id}")
    print("="*70)
    print(rec_df[["product_id", "product_name", "content_score", "price"]].to_string())
    print("="*70)

    engine.dispose()


if __name__ == "__main__":
    main()
