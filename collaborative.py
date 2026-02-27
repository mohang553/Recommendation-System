"""
recommendation_engine.py
Collaborative Filtering Recommendation Engine
Using Cosine Similarity on User-Item Rating Matrix

Can be:
  - Run standalone:  python recommendation_engine.py
  - Imported by FastAPI: from recommendation_engine import build_engine, load_model, get_recommendations
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from dotenv import load_dotenv
import os
import logging
from urllib.parse import quote_plus
import random

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = int(os.getenv("DB_PORT", 3306))
DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

TOP_N_SIMILAR_USERS   = 20
TOP_N_RECOMMENDATIONS = 10


# ── Build Engine ───────────────────────────────────────────────────────────────

def build_engine():
    password = quote_plus(DB_PASSWORD)  # safely encodes @ and special chars
    url = f"mysql+pymysql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True)


# ── Step 1: Load Ratings ───────────────────────────────────────────────────────

def load_ratings(engine) -> pd.DataFrame:
    """Load ratings from DB joined with product details."""
    logger.info("Loading ratings from database ...")
    query = """
        SELECT
            r.user_id,
            r.product_id,
            r.rating,
            p.title       AS product_title,
            p.stars       AS avg_stars,
            p.category_id
        FROM product_ratings r
        LEFT JOIN amazon_products p ON r.product_id = p.asin
    """
    df = pd.read_sql(query, engine)
    logger.info(f"  → {len(df):,} ratings | {df['user_id'].nunique():,} users | {df['product_id'].nunique():,} products")
    return df


# ── Step 2: Build User-Item Matrix ─────────────────────────────────────────────

def build_user_item_matrix(df: pd.DataFrame):
    """Build a users x products matrix. Empty cells = 0."""
    logger.info("Building user-item matrix ...")
    matrix = df.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        aggfunc="mean"
    ).fillna(0)
    sparse_matrix = csr_matrix(matrix.values)
    logger.info(f"  → Matrix shape: {matrix.shape[0]:,} users × {matrix.shape[1]:,} products")
    return matrix, sparse_matrix


# ── Step 3: Compute Cosine Similarity ─────────────────────────────────────────

def compute_user_similarity(sparse_matrix):
    """Compute cosine similarity between all users."""
    logger.info("Computing cosine similarity ...")
    similarity = cosine_similarity(sparse_matrix)
    logger.info(f"  → Similarity matrix: {similarity.shape}")
    return similarity


# ── Load Full Model (called by FastAPI on startup) ─────────────────────────────

def load_model(engine):
    """
    Load ratings, build matrix and similarity in one call.
    Called by FastAPI at startup to load model once into memory.
    Returns: (matrix, similarity_matrix)
    """
    ratings_df = load_ratings(engine)
    matrix, sparse_matrix = build_user_item_matrix(ratings_df)
    similarity_matrix = compute_user_similarity(sparse_matrix)
    logger.info("✓ Model loaded and ready")
    return matrix, similarity_matrix


# ── Step 4: Get Recommendations ───────────────────────────────────────────────

def get_recommendations(
    user_id: int,
    matrix: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_n: int = TOP_N_RECOMMENDATIONS
) -> pd.DataFrame:
    """
    Core recommendation logic.
    Finds top similar users → collects their ratings
    → filters already rated products → ranks by weighted score.
    Returns DataFrame with: product_id, predicted_rating, rank
    """
    if user_id not in matrix.index:
        logger.warning(f"User {user_id} not found in rating matrix!")
        return pd.DataFrame()

    user_idx          = matrix.index.get_loc(user_id)
    user_similarities = similarity_matrix[user_idx]

    # Top N similar users (exclude self at index 0)
    similar_indices  = np.argsort(user_similarities)[::-1][1:TOP_N_SIMILAR_USERS + 1]
    similar_user_ids = matrix.index[similar_indices]
    similar_scores   = user_similarities[similar_indices]

    # Products this user already rated — skip these
    already_rated = set(matrix.columns[matrix.loc[user_id] > 0])

    # Weighted score calculation
    weighted_scores = {}
    for sim_uid, sim_score in zip(similar_user_ids, similar_scores):
        if sim_score <= 0:
            continue
        sim_ratings    = matrix.loc[sim_uid]
        rated_products = sim_ratings[sim_ratings > 0]
        for product_id, rating in rated_products.items():
            if product_id in already_rated:
                continue
            if product_id not in weighted_scores:
                weighted_scores[product_id] = {"score": 0, "weight": 0}
            weighted_scores[product_id]["score"]  += sim_score * rating
            weighted_scores[product_id]["weight"] += sim_score

    if not weighted_scores:
        return pd.DataFrame()

    recs = [
        {
            "product_id":       pid,
            "predicted_rating": round(v["score"] / v["weight"], 3) if v["weight"] > 0 else 0,
        }
        for pid, v in weighted_scores.items()
    ]

    rec_df = (
        pd.DataFrame(recs)
        .sort_values("predicted_rating", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    rec_df["rank"] = range(1, len(rec_df) + 1)
    return rec_df


# ── Step 5: Enrich with Product Details ───────────────────────────────────────

def enrich_with_product_details(rec_df: pd.DataFrame, engine) -> pd.DataFrame:
    """Join recommendations with full product details from amazon_products."""
    if rec_df.empty:
        return rec_df
    pids       = ", ".join([f"'{p}'" for p in rec_df["product_id"].tolist()])
    details_df = pd.read_sql(f"""
        SELECT asin AS product_id, title AS product_name,
               stars AS avg_rating, reviews AS total_reviews,
               price, category_id
        FROM amazon_products WHERE asin IN ({pids})
    """, engine)
    return rec_df.merge(details_df, on="product_id", how="left")


# ── Step 6: Save Recommendations ──────────────────────────────────────────────

def save_recommendations(user_id: int, rec_df: pd.DataFrame, engine):
    """Save top recommendations for a single user into the recommendations table."""
    if rec_df.empty:
        return

    save_df = rec_df[["product_id", "predicted_rating", "rank"]].copy()
    save_df["user_id"]      = user_id
    save_df["generated_at"] = pd.Timestamp.now()

    save_df[["user_id", "product_id", "predicted_rating", "rank", "generated_at"]].to_sql(
        name="recommendations", con=engine,
        if_exists="append", index=False, chunksize=500, method="multi"
    )
    logger.info(f"  ✓ Saved recommendations for user {user_id}")


# ── Step 7: Batch Generate ────────────────────────────────────────────────────

def batch_generate(matrix: pd.DataFrame, similarity_matrix: np.ndarray, engine, sample_users: int = None):
    """Generate and save recommendations for all or a sampled set of users."""
    user_ids = matrix.index.tolist()
    if sample_users:
        user_ids = random.sample(user_ids, min(sample_users, len(user_ids)))
    logger.info(f"Batch generating for {len(user_ids):,} users ...")

    # Drop and recreate fresh recommendations table
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS recommendations"))
        conn.execute(text("""
    CREATE TABLE recommendations (
        id               INT AUTO_INCREMENT PRIMARY KEY,
        user_id          INT,
        product_id       VARCHAR(20),
        predicted_rating FLOAT,
        `rank`           INT,
        generated_at     DATETIME
    )
"""))
        conn.commit()

    success = 0
    for i, uid in enumerate(user_ids, 1):
        rec_df = get_recommendations(uid, matrix, similarity_matrix)
        if not rec_df.empty:
            save_recommendations(uid, rec_df, engine)
            success += 1
        if i % 100 == 0:
            logger.info(f"  Progress: {i:,} / {len(user_ids):,} ...")

    logger.info(f"  ✓ Batch done — {success:,} users saved")


# ── Main (standalone) ─────────────────────────────────────────────────────────

def main():
    engine = build_engine()

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("✓ Connected to RDS\n")

    # Load model
    matrix, similarity_matrix = load_model(engine)

    # Demo: show recommendations for first user
    sample_user_id = matrix.index[0]
    logger.info(f"\n── Recommendations for User {sample_user_id} ──")
    rec_df = get_recommendations(sample_user_id, matrix, similarity_matrix)
    rec_df = enrich_with_product_details(rec_df, engine)

    print("\n" + "="*70)
    print(f"  Top {TOP_N_RECOMMENDATIONS} Recommendations for User ID: {sample_user_id}")
    print("="*70)
    print(rec_df[["product_id", "product_name", "predicted_rating", "price"]].to_string())
    print("="*70)

    # Batch generate for 500 users and save to DB
    logger.info("\nStarting batch generation ...")
    batch_generate(matrix, similarity_matrix, engine, sample_users=500)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM recommendations")).scalar()
    logger.info(f"\n✓ Total recommendations in DB: {count:,}")

    engine.dispose()
    logger.info("All done ✓")


if __name__ == "__main__":
    main()