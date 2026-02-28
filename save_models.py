"""

Run this ONCE manually to build and save models to disk.
After this, main.py loads from disk instead of rebuilding.


"""

import pickle
import os
from collaborative import build_engine, load_model
from content_engine import load_content_model

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("🚀 Building and saving models — this runs once ...")

engine = build_engine()

# ── Save Collaborative Model ──────────────────────────────────────────────────
print("\n[1/2] Building collaborative model ...")
matrix, sim_matrix = load_model(engine)

with open(f"{MODELS_DIR}/cf_matrix.pkl",     "wb") as f: pickle.dump(matrix,     f)
with open(f"{MODELS_DIR}/cf_sim_matrix.pkl", "wb") as f: pickle.dump(sim_matrix, f)
print("✓ Collaborative model saved")

# ── Save Content Model ────────────────────────────────────────────────────────
print("\n[2/2] Building content model ...")
product_df, tfidf_matrix, product_index, vectorizer = load_content_model(engine)

with open(f"{MODELS_DIR}/product_df.pkl",     "wb") as f: pickle.dump(product_df,    f)
with open(f"{MODELS_DIR}/tfidf_matrix.pkl",   "wb") as f: pickle.dump(tfidf_matrix,  f)
with open(f"{MODELS_DIR}/product_index.pkl",  "wb") as f: pickle.dump(product_index, f)
with open(f"{MODELS_DIR}/vectorizer.pkl",     "wb") as f: pickle.dump(vectorizer,    f)
print("✓ Content model saved")

engine.dispose()
print("\n✅ All models saved to /models — run uvicorn main:app now")