"""
walk_forward_fragment.py
────────────────────────────────────────────────────────────────────────────
Revision-based walk-forward evaluation for fragment-level clone evolution
forecasting.

At each test revision r_t:
  - Train on ALL data from revisions strictly before r_t
  - Predict the label for each fragment that changed at r_t
  - Record metrics per-revision and globally

Usage
─────
  python ml/walk_forward_fragment.py \\
    --csv WorkFolder/Ctags/Datasets/CloneGenealogy/Type3_Block_fragment_dataset.csv
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
# Suppress sklearn config-propagation warnings emitted by joblib worker processes
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score,
)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


# ── CONFIG ───────────────────────────────────────────────────────────────────

MIN_TRAIN_ROWS = 30          # minimum training samples before first prediction
MIN_WARMUP_REVISIONS = 5     # skip the first N changed revisions for global warmup
REFIT_EVERY = 1              # retrain every N revisions (1 = always retrain)
CALIBRATION_WINDOW = 3       # last N predicted revisions for threshold tuning
THRESHOLD_DEFAULT = 0.5

# Features to exclude from model input
META_COLS = ["Revision", "gcid", "classid", "label"]


# ── MODEL FACTORY ────────────────────────────────────────────────────────────

def get_models():
    """Return dict of model_name -> unfitted model instance."""
    models = {}

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=10, is_unbalance=True,
            random_state=42, verbose=-1, n_jobs=-1,
        )

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_weight=5, use_label_encoder=False,
            eval_metric="logloss", random_state=42,
            verbosity=0, n_jobs=-1,
        )

    if HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=42,
            verbose=0,
        )

    return models


# ── METRICS ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute standard imbalanced-aware metrics."""
    m = {}
    m["n"] = len(y_true)
    m["n_pos"] = int(y_true.sum())
    m["n_neg"] = m["n"] - m["n_pos"]

    if len(np.unique(y_true)) < 2:
        # Only one class present
        m["mcc"] = 0.0
        m["auc_roc"] = 0.5
        m["pr_auc"] = m["n_pos"] / m["n"] if m["n"] > 0 else 0.0
        m["balanced_acc"] = 0.5
        m["f1"] = 0.0
        m["gmean"] = 0.0
        return m

    m["mcc"] = matthews_corrcoef(y_true, y_pred)
    m["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    m["f1"] = f1_score(y_true, y_pred, zero_division=0)

    if y_proba is not None:
        try:
            m["auc_roc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            m["auc_roc"] = 0.5
        try:
            m["pr_auc"] = average_precision_score(y_true, y_proba)
        except ValueError:
            m["pr_auc"] = 0.0
    else:
        m["auc_roc"] = 0.5
        m["pr_auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m["TP"] = int(tp)
    m["TN"] = int(tn)
    m["FP"] = int(fp)
    m["FN"] = int(fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    m["sensitivity"] = round(sens, 4)
    m["specificity"] = round(spec, 4)
    m["gmean"] = np.sqrt(sens * spec)

    return m


def tune_threshold(y_true, y_proba, n_points=50):
    """Find threshold maximizing MCC on calibration data."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return THRESHOLD_DEFAULT
    best_mcc = -2
    best_t = THRESHOLD_DEFAULT
    for t in np.linspace(0.1, 0.9, n_points):
        preds = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t
    return best_t


# ── WALK-FORWARD ENGINE ─────────────────────────────────────────────────────

def walk_forward(csv_path: str, output_dir: str = None):
    """Run revision-based walk-forward evaluation."""
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Rows: {len(df):,}, Features: {len(feature_cols)}, "
          f"Revisions: {df['Revision'].nunique()}")

    # Determine output directory — resolve relative to the script, not the CSV
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract system name from path for file naming
    csv_name = Path(csv_path).stem  # e.g. Type3_Block_fragment_dataset
    parts = csv_name.replace("_fragment_dataset", "").split("_")
    system = Path(csv_path).parent.parent.parent.name
    tag = f"{system}_{'_'.join(parts)}"

    # Sorted unique revisions
    unique_revisions = sorted(df["Revision"].unique())
    print(f"  Changed revisions: {len(unique_revisions)}")
    print(f"  First prediction after {MIN_WARMUP_REVISIONS} warmup revisions\n")

    models = get_models()
    print(f"Models: {list(models.keys())}\n")

    # Storage
    all_predictions = []       # per-prediction detail
    per_rev_metrics = []       # per-revision metrics per model
    global_y_true = defaultdict(list)
    global_y_pred = defaultdict(list)
    global_y_proba = defaultdict(list)

    # Calibration buffers (for threshold tuning)
    calib_y_true = defaultdict(list)
    calib_y_proba = defaultdict(list)

    n_predicted = 0

    for i, r_t in enumerate(unique_revisions):
        if i < MIN_WARMUP_REVISIONS:
            continue

        train_mask = df["Revision"] < r_t
        test_mask = df["Revision"] == r_t
        train = df[train_mask]
        test = df[test_mask]

        if len(train) < MIN_TRAIN_ROWS or len(test) == 0:
            continue

        X_train = train[feature_cols].values
        y_train = train["label"].values
        X_test = test[feature_cols].values
        y_test = test["label"].values

        # Skip if training has only one class
        if len(np.unique(y_train)) < 2:
            continue

        needs_refit = (n_predicted % REFIT_EVERY == 0) or n_predicted == 0

        for model_name, model in models.items():
            if needs_refit:
                try:
                    if model_name == "XGBoost":
                        # Set scale_pos_weight for imbalanced data
                        neg = (y_train == 0).sum()
                        pos = (y_train == 1).sum()
                        model.set_params(scale_pos_weight=neg / pos if pos > 0 else 1)
                    model.fit(X_train, y_train)
                except Exception as e:
                    print(f"  [WARN] {model_name} fit failed at rev {r_t}: {e}")
                    continue

            try:
                proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                continue

            # Determine threshold from calibration window
            cal_y = np.array(calib_y_true[model_name])
            cal_p = np.array(calib_y_proba[model_name])
            if len(cal_y) >= 10 and len(np.unique(cal_y)) >= 2:
                threshold = tune_threshold(cal_y, cal_p)
            else:
                threshold = THRESHOLD_DEFAULT

            preds = (proba >= threshold).astype(int)

            # Record per-prediction
            for j in range(len(test)):
                all_predictions.append({
                    "model": model_name,
                    "Revision": r_t,
                    "gcid": int(test.iloc[j]["gcid"]),
                    "y_true": int(y_test[j]),
                    "y_pred": int(preds[j]),
                    "y_proba": round(float(proba[j]), 4),
                    "threshold": round(threshold, 3),
                })

            # Per-revision metrics
            rev_m = compute_metrics(y_test, preds, proba)
            rev_m["model"] = model_name
            rev_m["Revision"] = r_t
            rev_m["threshold"] = round(threshold, 3)
            rev_m["train_size"] = len(train)
            per_rev_metrics.append(rev_m)

            # Update global accumulators
            global_y_true[model_name].extend(y_test.tolist())
            global_y_pred[model_name].extend(preds.tolist())
            global_y_proba[model_name].extend(proba.tolist())

            # Update calibration buffer (keep last CALIBRATION_WINDOW revisions worth)
            calib_y_true[model_name].extend(y_test.tolist())
            calib_y_proba[model_name].extend(proba.tolist())
            # Trim to last ~CALIBRATION_WINDOW * avg_per_rev samples
            max_calib = 200
            if len(calib_y_true[model_name]) > max_calib:
                calib_y_true[model_name] = calib_y_true[model_name][-max_calib:]
                calib_y_proba[model_name] = calib_y_proba[model_name][-max_calib:]

        n_predicted += 1

        # Progress
        if n_predicted % 25 == 0:
            # Show best model's rolling MCC
            best_mcc = -2
            best_name = ""
            for mn in models:
                if len(global_y_true[mn]) > 0:
                    mcc = matthews_corrcoef(global_y_true[mn], global_y_pred[mn])
                    if mcc > best_mcc:
                        best_mcc = mcc
                        best_name = mn
            print(f"  Rev {r_t} ({n_predicted} predicted) | "
                  f"Best: {best_name} MCC={best_mcc:.3f} ({len(global_y_true[best_name])} samples)")

    # ── GLOBAL METRICS ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GLOBAL RESULTS: {tag}")
    print(f"{'='*70}")

    global_results = []
    for model_name in models:
        yt = np.array(global_y_true[model_name])
        yp = np.array(global_y_pred[model_name])
        ypr = np.array(global_y_proba[model_name])
        if len(yt) == 0:
            continue
        gm = compute_metrics(yt, yp, ypr)
        gm["model"] = model_name
        gm["total_predictions"] = len(yt)
        global_results.append(gm)

        print(f"\n  {model_name}:")
        print(f"    MCC          = {gm['mcc']:.4f}")
        print(f"    AUC-ROC      = {gm['auc_roc']:.4f}")
        print(f"    PR-AUC       = {gm['pr_auc']:.4f}")
        print(f"    Balanced Acc = {gm['balanced_acc']:.4f}")
        print(f"    G-mean       = {gm['gmean']:.4f}")
        print(f"    F1           = {gm['f1']:.4f}")
        print(f"    Sensitivity  = {gm['sensitivity']:.4f}  (positive=dependent,  label=0)")
        print(f"    Specificity  = {gm['specificity']:.4f}  (negative=independent, label=1)")
        print(f"    TP={gm['TP']}  TN={gm['TN']}  FP={gm['FP']}  FN={gm['FN']}")
        print(f"    Predictions  = {gm['total_predictions']}")

    # ── SAVE RESULTS ─────────────────────────────────────────────────────────
    # Global metrics
    global_df = pd.DataFrame(global_results)
    global_path = output_dir / f"frag_wf_metrics_{tag}.csv"
    global_df.to_csv(global_path, index=False)
    print(f"\n  Global metrics -> {global_path}")

    # Per-revision metrics
    rev_df = pd.DataFrame(per_rev_metrics)
    rev_path = output_dir / f"frag_wf_per_revision_{tag}.csv"
    rev_df.to_csv(rev_path, index=False)
    print(f"  Per-revision    -> {rev_path}")

    # Per-prediction detail
    pred_df = pd.DataFrame(all_predictions)
    pred_path = output_dir / f"frag_wf_predictions_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions     -> {pred_path}")

    # ── MCC OVER REVISIONS PLOT ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 5))
        for model_name in models:
            model_rev = rev_df[rev_df["model"] == model_name].copy()
            if model_rev.empty:
                continue
            model_rev = model_rev.sort_values("Revision")
            # Rolling window MCC (window=10 revisions)
            model_rev["rolling_mcc"] = model_rev["mcc"].rolling(window=10, min_periods=1).mean()
            ax.plot(model_rev["Revision"], model_rev["rolling_mcc"],
                    label=model_name, alpha=0.8, linewidth=1.2)

        ax.set_xlabel("Revision")
        ax.set_ylabel("Rolling MCC (window=10)")
        ax.set_title(f"Fragment-Level Walk-Forward: MCC Over Revisions ({tag})")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plot_path = output_dir / f"frag_mcc_over_revisions_{tag}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  MCC plot        -> {plot_path}")
    except Exception as e:
        print(f"  [WARN] Could not create plot: {e}")

    # ── FEATURE IMPORTANCE (all models) ─────────────────────────────────────
    best_model_name = max(
        (mn for mn in models if global_y_true[mn]),
        key=lambda mn: matthews_corrcoef(
            global_y_true[mn], global_y_pred[mn]) if global_y_true[mn] else -2,
        default=None,
    )
    for model_name, model in models.items():
        try:
            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame({
                    "feature":    feature_cols,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)
                imp_path = output_dir / f"frag_feature_importance_{tag}_{model_name}.csv"
                imp.to_csv(imp_path, index=False)
                print(f"  Feature imp     -> {imp_path}")
        except Exception:
            pass

    if best_model_name and hasattr(models[best_model_name], "feature_importances_"):
        best_imp = pd.DataFrame({
            "feature":    feature_cols,
            "importance": models[best_model_name].feature_importances_,
        }).sort_values("importance", ascending=False)
        print(f"\n  Top 10 features ({best_model_name}, best by global MCC):")
        for _, row in best_imp.head(10).iterrows():
            print(f"    {row['feature']:38s} {row['importance']:.4f}")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Revision-based walk-forward evaluation for fragment-level clone forecasting")
    parser.add_argument("--csv", required=True,
                        help="Path to fragment_dataset.csv")
    parser.add_argument("--output", default=None,
                        help="Output directory for results (default: ml/results)")
    args = parser.parse_args()
    walk_forward(args.csv, args.output)
