"""
walk_forward.py
----------------------------------------------------------------------------
Walk-forward (prequential) evaluation of forecast models.

Evaluation protocol
-------------------
For each event index i in [MIN_TRAIN, n):
    train  = df.iloc[:i]            (all events strictly before i)
    test   = df.iloc[i:i+1]         (the single event at i)
    proba  = model.predict_proba(test)[:, 1]

To keep runtime reasonable, the model is refit every REFIT_EVERY events
(predictions in between use the last fitted model).

Threshold is tuned on a sliding CALIB_WINDOW of recent prior events,
so it adapts to local label-distribution drift instead of a single
threshold tuned on a fixed training set.

Why this matters
----------------
The single 70/30 split gave 91.8% positive in test and only 18 co-change
events.  Standard precision/recall/F1 are nearly indistinguishable from a
naive "predict all independent" baseline.  Walk-forward evaluation:
  * yields ~633 predictions over many revisions instead of 220
  * pulls co-change events from earlier revisions into the test stream
  * aggregates over many time points so metrics are statistically robust
  * matches deployment: at revision n predict using history 1..n-1

Outputs
-------
  ml/results/walk_forward_metrics.csv
  ml/results/walk_forward_predictions.csv
  ml/results/mcc_over_time.png
"""

import argparse
import os
# Must be set BEFORE sklearn/joblib import so workers inherit it.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import time
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from xgboost  import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, brier_score_loss,
    balanced_accuracy_score, confusion_matrix,
)

# Reuse helpers from the static-eval script
from train_test import (
    DEFAULT_CSV, RESULTS_DIR, TARGET, RANDOM_SEED,
    load, get_feature_cols, best_threshold,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────

MIN_TRAIN      = 100   # warmup: need this many events before first prediction
REFIT_EVERY    = 10    # retrain cadence (lower = slower, more responsive)
CALIB_WINDOW   = 50    # threshold-tuning window (recent events before i)
ROLLING_WINDOW = 50    # for the MCC-over-time plot


# ── MODELS ───────────────────────────────────────────────────────────────────

def model_factories(pos_weight: float) -> dict:
    """Factories so each fold gets a fresh, unfitted model."""
    return {
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, scale_pos_weight=pos_weight,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=pos_weight, random_state=RANDOM_SEED,
            n_jobs=-1, eval_metric="logloss", verbosity=0,
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED, verbose=0,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=500, max_depth=12, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }


# ── WALK-FORWARD CORE ────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, f_cols: list, target: str,
                 model_factory,
                 min_train: int = MIN_TRAIN,
                 refit_every: int = REFIT_EVERY,
                 calib_window: int = CALIB_WINDOW):
    """
    Returns (proba_pred, threshold_used, y_true, revisions, elapsed_seconds)
    for events in [min_train, n).
    """
    n = len(df)
    proba_pred = np.full(n, np.nan)
    threshold  = np.full(n, 0.5)
    model: Any = None
    last_good_threshold = 0.5

    t0 = time.time()
    for i in range(min_train, n):
        # Refit model on all prior events
        if (i - min_train) % refit_every == 0:
            model = model_factory()
            model.fit(df.iloc[:i][f_cols], df.iloc[:i][target])

        # Predict the current event
        proba_pred[i] = model.predict_proba(df.iloc[i:i+1][f_cols])[0, 1]

        # Tune threshold on the recent calibration window (keeps last good
        # threshold if window is single-class, which would make MCC=0 for all
        # thresholds and the scanner would pick a meaningless one).
        if i >= min_train + calib_window:
            cal = df.iloc[i - calib_window:i]
            if cal[target].nunique() > 1:
                cal_proba = model.predict_proba(cal[f_cols])[:, 1]
                last_good_threshold = best_threshold(
                    cal[target].values, cal_proba)
        threshold[i] = last_good_threshold

    elapsed = time.time() - t0
    mask = ~np.isnan(proba_pred)
    return (proba_pred[mask], threshold[mask],
            df[target].values[mask], df["Revision"].values[mask],
            df["gcid1"].values[mask], df["gcid2"].values[mask], elapsed)


# ── METRICS ──────────────────────────────────────────────────────────────────

def aggregate_metrics(y_true, proba, thresh) -> dict:
    pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    rec_pos = recall_score(y_true, pred, pos_label=1, zero_division=0)
    rec_neg = recall_score(y_true, pred, pos_label=0, zero_division=0)
    return {
        "MCC":            round(float(matthews_corrcoef(y_true, pred)), 4),
        "AUC-ROC":        round(float(roc_auc_score(y_true, proba)), 4),
        "PR-AUC":         round(float(average_precision_score(y_true, proba)), 4),
        "Balanced_Acc":   round(float(balanced_accuracy_score(y_true, pred)), 4),
        "G_mean":         round(float(np.sqrt(rec_pos * rec_neg)), 4),
        "Recall_pos":     round(float(rec_pos), 4),
        "Recall_neg":     round(float(rec_neg), 4),
        "Precision_pos":  round(float(precision_score(y_true, pred, pos_label=1, zero_division=0)), 4),
        "Precision_neg":  round(float(precision_score(y_true, pred, pos_label=0, zero_division=0)), 4),
        "F1":             round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "Brier":          round(float(brier_score_loss(y_true, proba)), 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def rolling_mcc(y_true: np.ndarray, pred: np.ndarray,
                window: int = ROLLING_WINDOW) -> np.ndarray:
    """MCC computed inside a rolling window of the prediction stream."""
    n = len(y_true)
    mccs = np.full(n, np.nan)
    for i in range(window, n + 1):
        y_w = y_true[i - window:i]
        p_w = pred[i - window:i]
        if len(np.unique(y_w)) > 1:
            mccs[i - 1] = matthews_corrcoef(y_w, p_w)
    return mccs


# ── PLOT ─────────────────────────────────────────────────────────────────────

def plot_rolling_mcc(results: dict, save_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, data in results.items():
        ax.plot(data["revisions"], data["rolling_mcc"],
                label=name, alpha=0.85, linewidth=1.5)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(f"Rolling MCC over time (window = {ROLLING_WINDOW} events)")
    ax.set_xlabel("Revision")
    ax.set_ylabel("MCC")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── PRINT TABLE ──────────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame):
    sep = "-" * 110
    cols = ["MCC", "AUC-ROC", "PR-AUC", "Balanced_Acc", "G_mean",
            "Recall_pos", "Recall_neg", "Precision_pos", "Precision_neg",
            "F1", "Brier", "N_predictions", "Time_sec"]
    print(f"\n{sep}")
    print("  WALK-FORWARD EVALUATION  (sorted by MCC)")
    print(sep)
    print(df[cols].to_string())
    print(sep)
    print("  Imbalance-aware (primary) : MCC, Balanced_Acc, G-mean")
    print("  Threshold-free            : AUC-ROC, PR-AUC")
    print("  G-mean = sqrt(recall_pos * recall_neg)  -> 0 if either class is ignored")
    print(sep)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main(csv_path: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    f_cols = get_feature_cols(df)

    pos = (df[TARGET] == 1).sum()
    neg = (df[TARGET] == 0).sum()
    pos_weight = neg / max(pos, 1)
    print(f"\nClass balance overall : {neg:,} co-change / {pos:,} independent")
    print(f"Pos_weight            : {pos_weight:.3f}")
    print(f"Min train / Refit / Calib window : "
          f"{MIN_TRAIN} / {REFIT_EVERY} / {CALIB_WINDOW}")

    factories = model_factories(pos_weight)
    runs = {}  # model_name -> dict with proba, thresh, y_true, revs, gc1, gc2, elapsed

    for name, factory in factories.items():
        print(f"\nWalk-forward [{name}] ...")
        proba, thresh, y_true, revs, gc1, gc2, elapsed = walk_forward(
            df, f_cols, TARGET, factory)
        runs[name] = dict(proba=proba, thresh=thresh, y_true=y_true,
                          revs=revs, gc1=gc1, gc2=gc2, elapsed=elapsed)
        print(f"  {elapsed:.1f}s | {len(y_true)} predictions")

    # ── Ensemble: average proba + average threshold across the boosters ──
    ENSEMBLE_MEMBERS = ["LightGBM", "XGBoost", "CatBoost"]
    if all(m in runs for m in ENSEMBLE_MEMBERS):
        ens_proba  = np.mean([runs[m]["proba"]  for m in ENSEMBLE_MEMBERS], axis=0)
        ens_thresh = np.mean([runs[m]["thresh"] for m in ENSEMBLE_MEMBERS], axis=0)
        ref = runs[ENSEMBLE_MEMBERS[0]]
        runs["Ensemble"] = dict(
            proba=ens_proba, thresh=ens_thresh,
            y_true=ref["y_true"], revs=ref["revs"],
            gc1=ref["gc1"], gc2=ref["gc2"],
            elapsed=sum(runs[m]["elapsed"] for m in ENSEMBLE_MEMBERS),
        )

    # ── Aggregate metrics, build prediction rows, build plot data ──
    metric_rows = []
    pred_rows = []
    plot_data = {}

    for name, r in runs.items():
        proba, thresh = r["proba"], r["thresh"]
        y_true, revs, gc1, gc2 = r["y_true"], r["revs"], r["gc1"], r["gc2"]
        pred = (proba >= thresh).astype(int)

        m = aggregate_metrics(y_true, proba, thresh)
        m["Model"] = name
        m["N_predictions"] = int(len(y_true))
        m["Time_sec"] = round(r["elapsed"], 1)
        metric_rows.append(m)

        plot_data[name] = {"revisions": revs, "rolling_mcc": rolling_mcc(y_true, pred)}

        for i in range(len(y_true)):
            pred_rows.append({
                "Model":     name,
                "Revision":  int(revs[i]),
                "gcid1":     int(gc1[i]),
                "gcid2":     int(gc2[i]),
                "y_true":    int(y_true[i]),
                "proba":     round(float(proba[i]), 4),
                "threshold": round(float(thresh[i]), 2),
                "pred":      int(pred[i]),
            })

        print(f"  [{name}]  MCC={m['MCC']}  AUC={m['AUC-ROC']}  "
              f"BalAcc={m['Balanced_Acc']}  G-mean={m['G_mean']}  "
              f"Recall(co)={m['Recall_neg']}  Recall(indep)={m['Recall_pos']}  "
              f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")

    metrics_df = (pd.DataFrame(metric_rows)
                    .set_index("Model")
                    .sort_values("MCC", ascending=False))
    pred_df = pd.DataFrame(pred_rows)

    metrics_df.to_csv(RESULTS_DIR / "walk_forward_metrics.csv")
    pred_df.to_csv(RESULTS_DIR / "walk_forward_predictions.csv", index=False)

    print_results(metrics_df)

    plot_rolling_mcc(plot_data, RESULTS_DIR / "mcc_over_time.png")
    print(f"\n  -> {RESULTS_DIR / 'walk_forward_metrics.csv'}")
    print(f"  -> {RESULTS_DIR / 'walk_forward_predictions.csv'}")
    print(f"  -> {RESULTS_DIR / 'mcc_over_time.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()
    main(args.csv)
