from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from collaborative import (
    build_engine,
    load_model,
    get_recommendations,
    enrich_with_product_details,
)
from content_engine import (
    load_content_model,
    get_content_recommendations,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── App State (loaded once at startup) ───────────────────────────────────────

class AppState:
    engine         = None
    matrix         = None
    sim_matrix     = None
    product_df     = None
    tfidf_matrix   = None
    product_index  = None

state = AppState()


# ── Lifespan: Load Both Models at Startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up — loading models ...")

    state.engine = build_engine()

    # Load collaborative filtering model
    logger.info("Loading collaborative filtering model ...")
    state.matrix, state.sim_matrix = load_model(state.engine)

    # Load content-based filtering model
    logger.info("Loading content-based filtering model ...")
    state.product_df, state.tfidf_matrix, state.product_index, _ = load_content_model(state.engine)

    logger.info("✓ Both models loaded and ready\n")
    yield

    # Shutdown
    logger.info("Shutting down — disposing DB engine ...")
    state.engine.dispose()


app = FastAPI(
    title="Recommendation Engine API",
    description="Test Collaborative Filtering and Content-Based Filtering endpoints",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def validate_user(user_id: int):
    """Raise 404 if user doesn't exist in the rating matrix."""
    if user_id not in state.matrix.index:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found in the rating matrix."
        )


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
def health_check():
    """Check if the API and both models are live."""
    return {
        "status"          : "ok",
        "users_loaded"    : int(state.matrix.shape[0]) if state.matrix is not None else 0,
        "products_loaded" : int(state.matrix.shape[1]) if state.matrix is not None else 0,
        "cf_model"        : state.sim_matrix  is not None,
        "content_model"   : state.tfidf_matrix is not None,
    }


# ── Endpoint 1: Collaborative Filtering ──────────────────────────────────────

@app.get("/recommend/collaborative/{user_id}", tags=["Collaborative Filtering"])
def collaborative_recommendations(
    user_id : int,
    top_n   : int  = Query(default=10, ge=1, le=50, description="Number of recommendations to return"),
    enrich  : bool = Query(default=True,             description="Include full product details")
):
    """
    Get recommendations using **Collaborative Filtering** (cosine similarity on user-item matrix).
    
    - Finds the most similar users to the given user
    - Recommends products those users rated highly that this user hasn't seen
    """
    validate_user(user_id)

    start = time.time()
    rec_df = get_recommendations(user_id, state.matrix, state.sim_matrix, top_n=top_n)

    if rec_df.empty:
        return JSONResponse(content={"user_id": user_id, "recommendations": [], "count": 0})

    if enrich:
        rec_df = enrich_with_product_details(rec_df, state.engine)

    elapsed = round(time.time() - start, 3)

    return {
        "user_id"        : user_id,
        "method"         : "collaborative_filtering",
        "count"          : len(rec_df),
        "response_time_s": elapsed,
        "recommendations": rec_df.to_dict(orient="records"),
    }


# ── Endpoint 2: Content-Based Filtering ──────────────────────────────────────

@app.get("/recommend/content/{user_id}", tags=["Content-Based Filtering"])
def content_recommendations(
    user_id     : int,
    top_n       : int = Query(default=10, ge=1, le=50, description="Number of recommendations to return"),
    top_rated_n : int = Query(default=5,  ge=1, le=20, description="Number of user's top-rated products to use as taste seeds"),
    enrich      : bool = Query(default=True,            description="Include full product details")
):
    """
    Get recommendations using **Content-Based Filtering** (TF-IDF cosine similarity).
    
    - Builds a taste profile from the user's highest-rated products
    - Finds unseen products most similar to that taste profile
    """
    validate_user(user_id)

    start  = time.time()
    rec_df = get_content_recommendations(
        user_id,
        state.matrix,
        state.tfidf_matrix,
        state.product_index,
        top_n=top_n,
        top_rated_n=top_rated_n
    )

    if rec_df.empty:
        return JSONResponse(content={"user_id": user_id, "recommendations": [], "count": 0})

    if enrich:
        rec_df = enrich_with_product_details(rec_df, state.engine)

    elapsed = round(time.time() - start, 3)

    return {
        "user_id"        : user_id,
        "method"         : "content_based_filtering",
        "count"          : len(rec_df),
        "response_time_s": elapsed,
        "recommendations": rec_df.to_dict(orient="records"),
    }


# ── Endpoint 3: Compare Both Side-by-Side ────────────────────────────────────

@app.get("/recommend/compare/{user_id}", tags=["Compare"])
def compare_recommendations(
    user_id : int,
    top_n   : int = Query(default=10, ge=1, le=50, description="Number of recommendations from each method")
):
    """
    Run **both engines** for the same user and return results side-by-side.
    Useful for directly comparing what each method recommends.
    """
    validate_user(user_id)

    start = time.time()

    cf_df = get_recommendations(user_id, state.matrix, state.sim_matrix, top_n=top_n)
    cb_df = get_content_recommendations(
        user_id, state.matrix, state.tfidf_matrix, state.product_index, top_n=top_n
    )

    if not cf_df.empty:
        cf_df = enrich_with_product_details(cf_df, state.engine)
    if not cb_df.empty:
        cb_df = enrich_with_product_details(cb_df, state.engine)

    # Products unique to each method
    cf_ids      = set(cf_df["product_id"]) if not cf_df.empty else set()
    cb_ids      = set(cb_df["product_id"]) if not cb_df.empty else set()
    overlap_ids = cf_ids & cb_ids

    elapsed = round(time.time() - start, 3)

    return {
        "user_id"                    : user_id,
        "response_time_s"            : elapsed,
        "overlap_count"              : len(overlap_ids),
        "overlap_product_ids"        : list(overlap_ids),
        "collaborative_filtering"    : cf_df.to_dict(orient="records") if not cf_df.empty else [],
        "content_based_filtering"    : cb_df.to_dict(orient="records") if not cb_df.empty else [],
    }


# ── Endpoint 4: List Sample Users ────────────────────────────────────────────

@app.get("/users/sample", tags=["Status"])
def sample_users(n: int = Query(default=10, ge=1, le=100, description="Number of sample user IDs to return")):
    """Returns a sample of user IDs from the matrix — useful for picking test users."""
    sample = state.matrix.index[:n].tolist()
    return {"sample_user_ids": sample, "total_users": len(state.matrix)}



@app.get("/search", tags=["Search"])
def search_products(
    q       : str   = Query(..., min_length=1, description="Search term e.g. 'shoes', 'laptop'"),
    top_n   : int   = Query(default=10, ge=1, le=50,  description="Number of results to return"),
    min_rating : float = Query(default=0.0, ge=0.0, le=5.0, description="Minimum average rating filter"),
):
    """
    Search products by name/title from the database.
    Results are sorted by average rating (highest first).
    """
    try:
        query = f"""
            SELECT 
                asin            AS product_id,
                title           AS product_name,
                stars           AS avg_rating,
                reviews         AS total_reviews,
                price,
                category_id
            FROM amazon_products
            WHERE title LIKE :search_term
              AND stars >= :min_rating
            ORDER BY stars DESC, reviews DESC
            LIMIT :top_n
        """

        from sqlalchemy import text
        with state.engine.connect() as conn:
            rows = conn.execute(text(query), {
                "search_term": f"%{q}%",
                "min_rating" : min_rating,
                "top_n"      : top_n
            }).fetchall()

        if not rows:
            return {
                "query"  : q,
                "count"  : 0,
                "results": [],
                "message": f"No products found for '{q}'"
            }

        results = [
            {
                "product_id"   : row.product_id,
                "product_name" : row.product_name,
                "avg_rating"   : row.avg_rating,
                "total_reviews": row.total_reviews,
                "price"        : row.price,
                "category_id"  : row.category_id,
            }
            for row in rows
        ]

        return {
            "query"  : q,
            "count"  : len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))