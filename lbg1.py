import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import lightgbm as lgb
import joblib


EMBED_DATASET_PATH = Path("embeddingdataset.json")


def load_embedding_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"{path} is empty")
        return json.loads(content)


def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "manual_score" not in df.columns:
        raise ValueError("manual_score not found in embeddingdataset.json")
    if "text_embedding" not in df.columns:
        raise ValueError("text_embedding not found in embeddingdataset.json")
    emb_array = np.vstack(df["text_embedding"].values)
    emb_dim = emb_array.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(emb_array, columns=emb_cols)
    df = df.drop(columns=["text_embedding"])
    df = pd.concat(
        [df.reset_index(drop=True), emb_df.reset_index(drop=True)],
        axis=1,
    )
    return df


def build_feature_matrix(df: pd.DataFrame):
    y = df["manual_score"].astype(float).values
    drop_cols = [
        "manual_score",
        "title",
        "summary",
        "categories",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    cat_cols = []
    if "primary_category" in X.columns:
        X["primary_category"] = X["primary_category"].astype("category")
        cat_cols.append("primary_category")
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        print(f"[WARN] Dropping unexpected object columns: {obj_cols}")
        X = X.drop(columns=obj_cols)
    cat_feature_indices = [
        X.columns.get_loc(c) for c in cat_cols if c in X.columns
    ]
    return X, y, cat_feature_indices


def split_data(X, y):
    bins = np.clip((y / 10).astype(int), 0, 10)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=bins,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=43,
        stratify=None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    cat_feature_indices,
):
    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_feature_indices if cat_feature_indices else None,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=cat_feature_indices if cat_feature_indices else None,
        reference=train_set,
        free_raw_data=False,
    )
    params = {
        "objective": "regression",
        "metric": ["rmse"],
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "verbose": -1,
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return model


def evaluate(model, X, y, split_name: str):
    preds = model.predict(X, num_iteration=model.best_iteration)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"[{split_name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


def show_feature_importance(model, feature_names, top_k: int = 40):
    importances = model.feature_importance(importance_type="gain")
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )
    print("\nTop features by gain:")
    print(fi.head(top_k).to_string(index=False))


def main():
    records = load_embedding_dataset(EMBED_DATASET_PATH)
    df = to_dataframe(records)
    X, y, cat_feature_indices = build_feature_matrix(df)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, y)
    print(
        f"Train size: {X_train.shape}, "
        f"Val size: {X_val.shape}, "
        f"Test size: {X_test.shape}"
    )
    model = train_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        cat_feature_indices,
    )
    evaluate(model, X_train, y_train, "TRAIN")
    evaluate(model, X_val, y_val, "VAL")
    evaluate(model, X_test, y_test, "TEST")
    show_feature_importance(model, X.columns.tolist(), top_k=50)
    model.save_model("lightgbmv1.txt")
    print("Saved model to lightgbmv1.txt")
    joblib.dump(model, "lightgbmv1.joblib")
    print("Saved model to lightgbmv1.joblib")


if __name__ == "__main__":
    main()
