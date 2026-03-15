import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
MODEL_PATH = Path("lightgbmv1.joblib")
DATASET_PATH = Path("embeddingdataset.json")

def load_data():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_record(rec):
    # This must match the feature extraction logic in lbg1.py
    scalar_features = ["author_count", "summary_len", "text_len", "years_since_published"]
    scalars = {k: float(rec.get(k, 0) or 0) for k in scalar_features}
    
    emb = rec.get("text_embedding")
    if emb is None:
        return None
        
    emb_cols = [f"emb_{i}" for i in range(len(emb))]
    
    # Create DataFrames
    scalar_df = pd.DataFrame([scalars])
    cat_df = pd.DataFrame({"primary_category": pd.Categorical([rec.get("primary_category", "unknown")])})
    emb_df = pd.DataFrame([emb], columns=emb_cols)
    
    # Concat
    X = pd.concat([scalar_df, cat_df, emb_df], axis=1)
    return X

def main():
    if not MODEL_PATH.exists():
        print(f"Error: {MODEL_PATH} not found. Run lbg1.py first.")
        return

    print(f"Loading model: {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    print(f"Loading dataset: {DATASET_PATH}...")
    data = load_data()
    
    # Pick the first record that has a score and embedding
    sample = None
    for rec in data:
        if rec.get("manual_score") and rec.get("text_embedding"):
            sample = rec
            break
            
    if not sample:
        print("No suitable sample found in dataset.")
        return
        
    print(f"\nTesting inference for ArXiv ID: {sample.get('arxiv_id')}")
    X = preprocess_record(sample)
    
    if X is None:
        print("Preprocessing failed.")
        return
        
    prediction = model.predict(X)[0]
    actual = sample.get("manual_score")
    
    print("-" * 30)
    print(f"Actual Score:    {actual}")
    print(f"Predicted Score: {prediction:.2f}")
    print(f"Difference:      {abs(actual - prediction):.2f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
