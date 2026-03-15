import json
import joblib
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Paths
MODEL_PATH = Path("lightgbmv1.joblib")
DATASET_PATH = Path("embeddingdataset.json")

def load_data():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_feature_matrix(records):
    rows = []
    cat_vals = []
    embs = []
    
    scalar_features = ["author_count", "summary_len", "text_len", "years_since_published"]
    
    for rec in records:
        emb = rec.get("text_embedding")
        if not emb: continue
        
        scalars = {k: float(rec.get(k, 0) or 0) for k in scalar_features}
        rows.append(scalars)
        cat_vals.append(rec.get("primary_category", "unknown"))
        embs.append(emb)
        
    if not rows: return None
    
    scalar_df = pd.DataFrame(rows)
    cat_df = pd.DataFrame({"primary_category": pd.Categorical(cat_vals)})
    emb_cols = [f"emb_{i}" for i in range(len(embs[0]))]
    emb_df = pd.DataFrame(embs, columns=emb_cols)
    
    X = pd.concat([scalar_df, cat_df, emb_df], axis=1)
    return X

def main():
    if not MODEL_PATH.exists():
        print(f"Error: {MODEL_PATH} not found.")
        return

    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    data = load_data()
    
    X = build_feature_matrix(data)
    if X is None:
        print("No data to test.")
        return
        
    num_records = len(X)
    iterations = 1000 // num_records + 1
    total_samples = iterations * num_records
    
    print(f"Starting throughput test with {total_samples} predictions...")
    
    start_time = time.time()
    for _ in range(iterations):
        _ = model.predict(X)
    end_time = time.time()
    
    total_time = end_time - start_time
    ips = total_samples / total_time
    
    print("-" * 30)
    print(f"Total Time:      {total_time:.4f} seconds")
    print(f"Total Samples:   {total_samples}")
    print(f"Throughput:      {ips:.2f} inferences/sec")
    print(f"Latency:         {(total_time/total_samples)*1000:.4f} ms per sample")
    print("-" * 30)

if __name__ == "__main__":
    main()
