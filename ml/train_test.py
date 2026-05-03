"""
train_test.py
────────────────────────────────────────────────────────────────────────────
Train and evaluate 4 classifiers on the forecast_dataset.

Split strategy  (SPLIT_BY_ROWS = True, recommended):
    Sort all rows globally by Revision.
    First 70% of rows  → TRAIN  (earliest events)
    Last  30% of rows  → TEST   (most-recent events)
    This guarantees the model always trains on events that happened before
    the events it is tested on, and gives sufficient training data even
    when change events are sparse in early revisions.

    Alternative (SPLIT_BY_ROWS = False):
    Cut at 70% of the revision NUMBER range.
    Strict but can give very few training rows if most events cluster late.

At each test row at revision n:
    features  = state of the pair at the previous event  (history 1..n-1)
    label     = type of change AT revision n  (0=co-change, 1=independent)

Models:
    LightGBM, XGBoost, CatBoost, RandomForest

Primary metrics : MCC, AUC-ROC, PR-AUC
"""

import argparse
import os
# Must be set BEFORE sklearn/joblib import so workers inherit it.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import joblib
from typing import Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from lightgbm import LGBMClassifier
from xgboost  import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, brier_score_loss,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────

DEFAULT_CSV = (
    r"WorkFolder\Ctags\Datasets\CloneGenealogy"
    r"\Type3_Block_forecast_dataset.csv"
)

RESULTS_DIR  = Path(__file__).parent / "results"
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
TARGET       = "will_diverge"
SPLIT_BY_ROWS = True   # True = row-count split (recommended)
                       # False = revision-range split (strict temporal)

NON_FEATURES = {
    TARGET, "Revision",
    "gcid1", "gcid2", "classid",
    "wies", "will_independently_evolve",
    "will_diverge", "will_diverge_decay",  # both labels always excluded from features
}

RUN_PREFIX = "run"  # default; overridden from __main__ or by caller

def extract_run_prefix(csv_path: str) -> str:
    """
    Derive a '{System}_{CloneType}' prefix from the CSV path.

    Examples:
        'WorkFolder/Jmol/Datasets/.../Type3_Block_forecast_dataset.csv'  -> 'Jmol_Type3'
        'WorkFolder/Ctags/Datasets/.../Type1_Block_evolution_dataset.csv' -> 'Ctags_Type1'
    """
    p = Path(csv_path)
    # System name: folder between 'WorkFolder' and 'Datasets'
    parts = p.parts
    system = "Unknown"
    for i, part in enumerate(parts):
        if part.lower() == "workfolder" and i + 1 < len(parts):
            system = parts[i + 1]
            break

    # Clone type: 'Type1', 'Type2', 'Type3' from the filename
    stem = p.stem  # e.g. 'Type3_Block_forecast_dataset'
    clone_type = stem.split("_")[0] if "_" in stem else stem  # 'Type3'
    return f"{system}_{clone_type}"

# ── DATA ─────────────────────────────────────────────────────────────────────

def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Always sort globally by Revision so rows are in temporal order
    df = df.sort_values("Revision").reset_index(drop=True)
    print(f"Loaded  {len(df):,} rows x {df.shape[1]} cols")
    print(f"Revision range  : {df['Revision'].min()} - {df['Revision'].max()}")
    vc = df[TARGET].value_counts()
    print(f"Label balance   : {vc.get(0,0):,} co-change / {vc.get(1,0):,} independent")
    return df

# ── TEMPORAL SPLIT ───────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame):
    """
    SPLIT_BY_ROWS = True:
        First 70% of rows (sorted by Revision) → TRAIN
        Last  30% of rows                      → TEST
        Guarantees temporal order and sufficient training data.

    SPLIT_BY_ROWS = False:
        Cut at 70% of revision NUMBER range → strict but may give tiny train set.
    """
    if SPLIT_BY_ROWS:
        cut_idx = int(len(df) * TRAIN_RATIO)
        train = df.iloc[:cut_idx].copy()
        test  = df.iloc[cut_idx:].copy()
        cut_label = f"row {cut_idx}  (revision {df.iloc[cut_idx]['Revision']:.0f})"
    else:
        rev_min = df["Revision"].min()
        rev_max = df["Revision"].max()
        cut_rev = rev_min + TRAIN_RATIO * (rev_max - rev_min)
        train = df[df["Revision"] <= cut_rev].copy()
        test  = df[df["Revision"] >  cut_rev].copy()
        cut_label = f"revision {cut_rev:.0f}"

    print(f"\nSplit at {cut_label}  ({TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%})")
    print(f"  TRAIN : {len(train):>6,} rows | positive rate {train[TARGET].mean():.1%}")
    print(f"  TEST  : {len(test):>6,}  rows | positive rate {test[TARGET].mean():.1%}")
    return train, test

# ── FEATURES ─────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c not in NON_FEATURES]
    print(f"\nFeature count   : {len(cols)}")
    return cols

# ── MODELS ───────────────────────────────────────────────────────────────────

def build_models(pos_weight: float) -> dict:
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, scale_pos_weight=pos_weight,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=pos_weight, random_state=RANDOM_SEED,
            n_jobs=-1, eval_metric="logloss", verbosity=0,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED, verbose=0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=12, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }

# ── THRESHOLD TUNING ─────────────────────────────────────────────────────────

def best_threshold(y_true, proba, metric="mcc") -> float:
    """Find the decision threshold that maximises MCC on the given set."""
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        pred = (proba >= t).astype(int)
        if metric == "mcc":
            score = float(matthews_corrcoef(y_true, pred))
        else:
            score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_score:
            best_score, best_t = score, t
    return best_t

# ── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate(name, model: Any, X_tr, y_tr, X_te, y_te) -> dict:
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    # Tune threshold on the last 20% of training rows (temporally closest to
    # test — its label distribution is closer to the test distribution than
    # using the whole training set, which gives a better-calibrated threshold).
    val_cut = int(len(X_tr) * 0.80)
    val_proba = model.predict_proba(X_tr.iloc[val_cut:])[:, 1]
    t = best_threshold(y_tr.iloc[val_cut:], val_proba, metric="mcc")
    pred = (proba >= t).astype(int)

    # Oracle threshold: best achievable MCC on the test set itself.
    # Only valid for research/thesis comparison — not usable in deployment.
    t_oracle = best_threshold(y_te, proba, metric="mcc")
    pred_oracle = (proba >= t_oracle).astype(int)

    return {
        "Model":      name,
        "Threshold":  round(float(t), 2),
        "MCC":        round(float(matthews_corrcoef(y_te, pred)), 4),
        "MCC_oracle": round(float(matthews_corrcoef(y_te, pred_oracle)), 4),
        "T_oracle":   round(float(t_oracle), 2),
        "AUC-ROC":    round(float(roc_auc_score(y_te, proba)), 4),
        "PR-AUC":     round(float(average_precision_score(y_te, proba)), 4),
        "F1":         round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "Precision":  round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "Recall":     round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "Brier":      round(float(brier_score_loss(y_te, proba)), 4),
        "_model":     model,
        "_proba":     proba,
        "_pred":      pred,
    }

# ── CROSS-VALIDATION (TimeSeriesSplit) ───────────────────────────────────────

def cross_validate(name, model_cls: Any, kwargs, X, y, n_splits=5):
    """
    Walk-forward cross-validation: each fold trains on all prior rows
    and tests on the next block.  Rows must be sorted by Revision.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mccs, aucs = [], []
    for tr_idx, te_idx in tscv.split(X):
        m = model_cls(**kwargs)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = m.predict_proba(X.iloc[te_idx])[:, 1]
        val_cut = int(len(tr_idx) * 0.80)
        val_proba = m.predict_proba(X.iloc[tr_idx[val_cut:]])[:, 1]
        t   = best_threshold(y.iloc[tr_idx[val_cut:]], val_proba)
        pred = (proba >= t).astype(int)
        mccs.append(float(matthews_corrcoef(y.iloc[te_idx], pred)))
        aucs.append(float(roc_auc_score(y.iloc[te_idx], proba)))
    return np.mean(mccs), np.std(mccs), np.mean(aucs), np.std(aucs)

# ── PLOTS ────────────────────────────────────────────────────────────────────

def plot_shap(model, X_test: pd.DataFrame):
    print("\nComputing SHAP (LightGBM) ...")
    explainer   = shap.TreeExplainer(model)
    sv          = explainer.shap_values(X_test)
    sv          = sv[1] if isinstance(sv, list) else sv

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test, show=False, max_display=20)
    plt.tight_layout()
    out = RESULTS_DIR / f"{RUN_PREFIX}_shap_summary.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  -> {out}")

def plot_importance(model, cols: list):
    imp = pd.Series(model.feature_importances_, index=cols).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("LightGBM - Feature Importance (top 20)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    out = RESULTS_DIR / f"{RUN_PREFIX}_feature_importance.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  -> {out}")

def plot_label_drift(df: pd.DataFrame, cut_idx: int):
    """Show positive rate across revision buckets — reveals temporal drift."""
    df = df.copy()
    df["bucket"] = pd.qcut(df["Revision"], q=10, duplicates="drop")
    drift = df.groupby("bucket", observed=True)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    drift.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(x=len(drift) * TRAIN_RATIO - 0.5, color="red",
               linestyle="--", label="train/test cut")
    ax.set_title("Positive rate (independent changes) by revision bucket")
    ax.set_ylabel("Fraction independent")
    ax.set_xlabel("Revision bucket")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / f"{RUN_PREFIX}_label_drift.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  -> {out}")

# ── RESULTS TABLE ─────────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame):
    sep = "-" * 90
    display_cols = ["Threshold","MCC","MCC_oracle","T_oracle",
                    "AUC-ROC","PR-AUC","F1","Precision","Recall","Brier"]
    print(f"\n{sep}")
    print("  MODEL COMPARISON  (sorted by MCC, threshold tuned on last 20% of train)")
    print("  MCC_oracle = best possible MCC at optimal test threshold (upper bound)")
    print(sep)
    print(df[display_cols].to_string())
    print(sep)
    print("  Primary : AUC-ROC, PR-AUC, MCC_oracle  (higher = better)")
    print("  Brier   : lower = better")
    print(sep)

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main(csv_path: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df     = load(csv_path)
    train, test = temporal_split(df)
    f_cols = get_feature_cols(df)

    X_train, y_train = train[f_cols], train[TARGET]
    X_test,  y_test  = test[f_cols],  test[TARGET]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"Imbalance ratio : {pos_weight:.2f}  "
          f"(neg={int((y_train==0).sum())}, pos={int((y_train==1).sum())})")

    # Naive baseline for reference
    naive_f1  = float(f1_score(y_test, np.ones(len(y_test), dtype=int), zero_division=0))
    naive_mcc = float(matthews_corrcoef(y_test, np.ones(len(y_test), dtype=int)))
    print(f"\nNaive baseline (predict all independent):  "
          f"MCC={naive_mcc:.4f}  F1={naive_f1:.4f}")

    models  = build_models(pos_weight)
    results = []
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        row = evaluate(name, model, X_train, y_train, X_test, y_test)
        trained[name] = row.pop("_model")
        _ = row.pop("_proba"); _ = row.pop("_pred")
        results.append(row)
        print(f"  threshold={row['Threshold']}  MCC={row['MCC']:>7}  "
              f"oracle_MCC={row['MCC_oracle']:>7}  "
              f"AUC={row['AUC-ROC']:>7}  PR-AUC={row['PR-AUC']:>7}")

    results_df = (
        pd.DataFrame(results).set_index("Model")
                             .sort_values("MCC", ascending=False)
    )
    print_results(results_df)
    results_df.to_csv(RESULTS_DIR / f"{RUN_PREFIX}_model_comparison.csv")

    # Walk-forward cross-validation on LightGBM
    print("\nWalk-forward CV (LightGBM, 5 folds) ...")
    lgb_kwargs = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                      num_leaves=31, scale_pos_weight=pos_weight,
                      random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    mcc_mean, mcc_std, auc_mean, auc_std = cross_validate(
        "LightGBM", LGBMClassifier, lgb_kwargs,
        df[f_cols], df[TARGET]
    )
    print(f"  MCC  : {mcc_mean:.4f} +/- {mcc_std:.4f}")
    print(f"  AUC  : {auc_mean:.4f} +/- {auc_std:.4f}")

    # Save model
    joblib.dump(trained["LightGBM"], RESULTS_DIR / f"{RUN_PREFIX}_trained_lgbm.pkl")
    print(f"\nSaved model -> {RESULTS_DIR / f'{RUN_PREFIX}_trained_lgbm.pkl'}")

    # Plots
    plot_shap(trained["LightGBM"], X_test)
    plot_importance(trained["LightGBM"], f_cols)
    plot_label_drift(df, int(len(df) * TRAIN_RATIO))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--target", default="will_diverge",
                        help="Target column. Use 'will_diverge_decay' for Option C.")
    args = parser.parse_args()
    TARGET = args.target
    RUN_PREFIX = extract_run_prefix(args.csv)
    # Re-derive NON_FEATURES with the chosen target
    NON_FEATURES = {
        TARGET, "Revision",
        "gcid1", "gcid2", "classid",
        "wies", "will_independently_evolve",
        "will_diverge", "will_diverge_decay",
    }
    main(args.csv)
