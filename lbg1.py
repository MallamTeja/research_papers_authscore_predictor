"""
lbg1.py — Train a LightGBM regressor to predict manual_score.

Input:  embeddingdataset.json  (merged dataset.json + dataset3.json data,
        with text replaced by text_embedding vectors, all scored records only)
Output: shap_summary.png       (SHAP beeswarm — feature importance)
        shap_bar.png           (SHAP mean |value| bar chart)
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
EMBED_DATASET_PATH = Path("embeddingdataset.json")
SHAP_SUMMARY_PNG   = Path("shap_summary.png")
SHAP_BAR_PNG       = Path("shap_bar.png")

# ── Feature config ─────────────────────────────────────────────────────────────
# Scalar features from dataset schema (alongside the 384-dim embedding)
SCALAR_FEATURES = [
    "author_count",
    "summary_len",
    "text_len",
    "years_since_published",
]

# Categorical feature — will be label-encoded
CAT_FEATURE = "primary_category"

# Target
TARGET = "manual_score"

# Sub-scores to drop (they leak the target; the model should predict from content only)
LEAK_COLS = ["novelty_score", "rigor_score", "impact_score", "manual_score"]


# ── I/O ────────────────────────────────────────────────────────────────────────
def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run dc.py → distillation.py → emb.py first to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.loads(f.read().strip())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} is empty or not a list.")
    return data


# ── Feature extraction ─────────────────────────────────────────────────────────
def build_features(records: list[dict]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Returns:
        X         — DataFrame with scalar + embedding columns
        y         — 1-D numpy array of manual_score
        feat_names — ordered column names (same order as X)
    """
    rows = []
    targets = []
    cat_vals = []

    for rec in records:
        emb = rec.get("text_embedding")
        if emb is None or not isinstance(emb, list):
            print(f"  [SKIP] {rec.get('arxiv_id','?')} — missing text_embedding")
            continue

        score = rec.get(TARGET, 0)
        if score == 0:
            # unscored record — skip (shouldn't be in embeddingdataset but guard anyway)
            continue

        scalars = {k: float(rec.get(k, 0) or 0) for k in SCALAR_FEATURES}
        scalars["emb"] = emb  # placeholder — expanded below
        rows.append(scalars)
        targets.append(float(score))
        cat_vals.append(str(rec.get(CAT_FEATURE, "unknown")))

    if not rows:
        raise ValueError(
            "No valid scored records with embeddings found in embeddingdataset.json.\n"
            "Make sure distillation.py has run and emb.py has processed scored records."
        )

    # Build embedding columns
    emb_array = np.array([r.pop("emb") for r in rows], dtype=np.float32)
    emb_dim   = emb_array.shape[1]
    emb_cols  = [f"emb_{i}" for i in range(emb_dim)]

    scalar_df = pd.DataFrame(rows)  # author_count, summary_len, text_len, years_since_published
    emb_df    = pd.DataFrame(emb_array, columns=emb_cols)
    cat_df    = pd.DataFrame({"primary_category": pd.Categorical(cat_vals)})

    X = pd.concat([scalar_df.reset_index(drop=True),
                   cat_df.reset_index(drop=True),
                   emb_df.reset_index(drop=True)], axis=1)
    y = np.array(targets, dtype=np.float32)

    return X, y, X.columns.tolist()


# ── Training ───────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "regression_l1",   # MAE loss — more robust to score outliers
    "metric":           ["rmse", "mae"],
    "learning_rate":    0.02,
    "num_leaves":       31,
    "max_depth":        -1,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_data_in_leaf": 5,               # small — dataset is small right now
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":          -1,
    "n_jobs":           -1,
}


def train_model(X_train, y_train, X_val, y_val, cat_cols: list[str]) -> lgb.Booster:
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]

    train_set = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_indices or "auto",
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=cat_indices or "auto",
        reference=train_set,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=150, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        LGBM_PARAMS,
        train_set,
        num_boost_round=3000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model: lgb.Booster, X, y, split_name: str) -> None:
    preds = model.predict(X, num_iteration=model.best_iteration)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    print(f"  [{split_name:5s}]  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")


# ── SHAP explanation ───────────────────────────────────────────────────────────
def run_shap(model: lgb.Booster, X: pd.DataFrame, feat_names: list[str]) -> None:
    print("\nRunning SHAP analysis...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Readable names: abbreviate embedding dims, keep scalar/cat names intact
    display_names = []
    for n in feat_names:
        if n.startswith("emb_"):
            display_names.append(n)  # kept for completeness; grouped below
        else:
            display_names.append(n)

    # ── Bar plot: top-20 mean |SHAP| (grouped: all emb_* as "embedding (384d)") ──
    mean_abs = np.abs(shap_values).mean(axis=0)
    fi_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})

    # Group all embedding dims into one synthetic row
    emb_mask    = fi_df["feature"].str.startswith("emb_")
    emb_total   = fi_df.loc[emb_mask, "mean_abs_shap"].sum()
    non_emb_df  = fi_df.loc[~emb_mask].copy()
    emb_row     = pd.DataFrame([{"feature": "text_embedding (384d total)", "mean_abs_shap": emb_total}])
    fi_grouped  = pd.concat([non_emb_df, emb_row], ignore_index=True)
    fi_grouped  = fi_grouped.sort_values("mean_abs_shap", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#7F77DD" if f == "text_embedding (384d total)" else "#1D9E75"
              for f in fi_grouped["feature"]]
    ax.barh(fi_grouped["feature"], fi_grouped["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP value|  (impact on manual_score prediction)")
    ax.set_title("Feature importance — LightGBM paper scorer")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(SHAP_BAR_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved: {SHAP_BAR_PNG}")

    # ── Beeswarm: top-15 individual features (skip emb_ dims unless they dominate) ──
    non_emb_indices = [i for i, n in enumerate(feat_names) if not n.startswith("emb_")]
    top_emb_indices = np.argsort(mean_abs)[::-1][:5]        # top 5 embedding dims
    keep = sorted(set(non_emb_indices) | set(top_emb_indices.tolist()))[:20]

    X_sub     = X.iloc[:, keep]
    sv_sub    = shap_values[:, keep]
    sub_names = [feat_names[i] for i in keep]

    # Rename top embedding dims for readability
    sub_names = [f"emb_dim_{n.split('_')[1]}" if n.startswith("emb_") else n
                 for n in sub_names]

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        sv_sub,
        X_sub,
        feature_names=sub_names,
        show=False,
        plot_size=None,
        max_display=20,
    )
    plt.title("SHAP beeswarm — top features shaping paper score")
    plt.tight_layout()
    fig2.savefig(SHAP_SUMMARY_PNG, dpi=150)
    plt.close(fig2)
    print(f"  Saved: {SHAP_SUMMARY_PNG}")

    # ── Print top non-embedding features ──
    print("\nTop non-embedding features by mean |SHAP|:")
    non_emb_df_sorted = fi_df.loc[~emb_mask].sort_values("mean_abs_shap", ascending=False)
    for _, row in non_emb_df_sorted.iterrows():
        print(f"  {row['feature']:30s}  {row['mean_abs_shap']:.4f}")
    print(f"\n  text_embedding (384d combined):  {emb_total:.4f}")


# ── Cross-validation (for small dataset confidence) ───────────────────────────
def cross_validate(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str], k: int = 5) -> None:
    print(f"\nRunning {k}-fold cross-validation...")
    kf   = KFold(n_splits=k, shuffle=True, random_state=42)
    maes = []
    cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        tr_set = lgb.Dataset(X_tr, label=y_tr,
                             categorical_feature=cat_indices or "auto",
                             free_raw_data=False)
        va_set = lgb.Dataset(X_va, label=y_va,
                             categorical_feature=cat_indices or "auto",
                             reference=tr_set, free_raw_data=False)

        m = lgb.train(
            LGBM_PARAMS, tr_set, num_boost_round=1000,
            valid_sets=[va_set], valid_names=["val"],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(period=9999)],
        )
        preds = m.predict(X_va, num_iteration=m.best_iteration)
        mae   = mean_absolute_error(y_va, preds)
        maes.append(mae)
        print(f"  Fold {fold}: MAE={mae:.3f}")

    print(f"  CV MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Load
    print(f"Loading {EMBED_DATASET_PATH}...")
    records = load_dataset(EMBED_DATASET_PATH)
    print(f"  Total records in file: {len(records)}")

    # 2. Build features
    X, y, feat_names = build_features(records)
    print(f"  Scored records used for training: {len(y)}")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Score range: {y.min():.0f} – {y.max():.0f}  mean={y.mean():.1f}")

    if len(y) < 10:
        print(
            "\n[WARNING] Only {len(y)} scored records — model will overfit badly.\n"
            "Run distillation.py on more records first for a meaningful model.\n"
            "Proceeding anyway so you have a working artifact to iterate on.\n"
        )

    cat_cols = [CAT_FEATURE] if CAT_FEATURE in X.columns else []

    # 3. Cross-validate (gives honest estimate on small dataset)
    if len(y) >= 5:
        cross_validate(X, y, cat_cols, k=min(5, len(y) // 2))

    # 4. Train/val split for final model
    if len(y) >= 4:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # Too few records — train on everything, validate on everything
        X_train, X_val, y_train, y_val = X, X, y, y

    print(f"\nFinal model training — train={len(y_train)}  val={len(y_val)}")
    model = train_model(X_train, y_train, X_val, y_val, cat_cols)

    # 5. Evaluate
    print("\nEvaluation:")
    evaluate(model, X_train, y_train, "TRAIN")
    evaluate(model, X_val,   y_val,   "VAL")
    evaluate(model, X,       y,       "ALL")

    # 6. SHAP
    run_shap(model, X, feat_names)

    # 7. Quick sanity: predict on the training records and print side-by-side
    preds_all = model.predict(X, num_iteration=model.best_iteration)
    print("\nPrediction sanity check (first 10 scored records):")
    print(f"  {'Actual':>8}  {'Predicted':>10}")
    for actual, pred in zip(y[:10], preds_all[:10]):
        print(f"  {actual:8.0f}  {pred:10.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()