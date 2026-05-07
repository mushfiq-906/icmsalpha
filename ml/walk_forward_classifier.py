"""
walk_forward_classifier.py
────────────────────────────────────────────────────────────────────────────
Revision-based walk-forward evaluator for the per-fragment classifier
("dependent vs independent" methodology).

At each verification revision r_t (the revision where one or more fragments
actually changed):
  - Train on ALL rows with verification_revision < r_t
  - Predict the label for each row with verification_revision == r_t
  - Record per-prediction detail (including decision_index for trajectory plot)
  - Record per-revision and global metrics

Label convention from build_classifier_dataset.py:
    label = 1  ->  dependent    (positive class)
    label = 0  ->  independent

Usage
─────
  python ml/walk_forward_classifier.py \\
    --csv WorkFolder/Ctags/Datasets/CloneGenealogy/Type3_Block_classifier_dataset.csv
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

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


MIN_TRAIN_ROWS = 30
MIN_WARMUP_REVISIONS = 5
REFIT_EVERY = 1
CALIBRATION_WINDOW_MAX = 200
THRESHOLD_DEFAULT = 0.5

META_COLS = [
    "verification_revision", "gcid", "classid", "decision_index",
    "prev_change_revision", "gap_since_prev_change", "label",
]


def get_models():
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


def compute_metrics(y_true, y_pred, y_proba=None):
    m = {}
    m["n"] = len(y_true)
    m["n_pos"] = int(np.asarray(y_true).sum())
    m["n_neg"] = m["n"] - m["n_pos"]

    if len(np.unique(y_true)) < 2:
        m["mcc"] = 0.0
        m["auc_roc"] = 0.5
        m["pr_auc"] = m["n_pos"] / m["n"] if m["n"] > 0 else 0.0
        m["balanced_acc"] = 0.5
        m["f1"] = 0.0
        m["TP"] = m["TN"] = m["FP"] = m["FN"] = 0
        m["sensitivity"] = 0.0
        m["specificity"] = 0.0
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
    m["gmean"] = float(np.sqrt(sens * spec))

    return m


def tune_threshold(y_true, y_proba, n_points=50):
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


def derive_tag(csv_path: Path) -> str:
    name = csv_path.stem  # e.g. Type3_Block_classifier_dataset
    parts = name.replace("_classifier_dataset", "").split("_")
    # Walk up: .../WorkFolder/{System}/Datasets/CloneGenealogy/<csv>
    system = csv_path.parent.parent.parent.name if len(csv_path.parents) >= 3 else "system"
    return f"{system}_{'_'.join(parts)}"


def walk_forward(csv_path: str, output_dir: str = None,
                 min_warmup_revisions: int = MIN_WARMUP_REVISIONS,
                 refit_every: int = REFIT_EVERY,
                 min_train_rows: int = MIN_TRAIN_ROWS):
    csv_path = Path(csv_path)
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Rows: {len(df):,}, Features: {len(feature_cols)}, "
          f"Verification revisions: {df['verification_revision'].nunique()}")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = derive_tag(csv_path)

    unique_revisions = sorted(df["verification_revision"].unique())
    print(f"  Verification revisions: {len(unique_revisions)}")
    print(f"  First prediction after {min_warmup_revisions} warmup revisions\n")

    models = get_models()
    print(f"Models: {list(models.keys())}\n")

    all_predictions = []
    per_rev_metrics = []
    global_y_true = defaultdict(list)
    global_y_pred = defaultdict(list)
    global_y_proba = defaultdict(list)

    calib_y_true = defaultdict(list)
    calib_y_proba = defaultdict(list)

    n_predicted = 0
    last_fitted_models = {}

    for i, r_t in enumerate(unique_revisions):
        if i < min_warmup_revisions:
            continue

        train = df[df["verification_revision"] < r_t]
        test = df[df["verification_revision"] == r_t]

        if len(train) < min_train_rows or len(test) == 0:
            continue

        X_train = train[feature_cols].values
        y_train = train["label"].values
        X_test = test[feature_cols].values
        y_test = test["label"].values

        if len(np.unique(y_train)) < 2:
            continue

        needs_refit = (n_predicted % refit_every == 0) or n_predicted == 0

        for model_name, model in models.items():
            if needs_refit:
                try:
                    if model_name == "XGBoost":
                        neg = (y_train == 0).sum()
                        pos = (y_train == 1).sum()
                        model.set_params(scale_pos_weight=neg / pos if pos > 0 else 1)
                    model.fit(X_train, y_train)
                    last_fitted_models[model_name] = model
                except Exception as e:
                    print(f"  [WARN] {model_name} fit failed at rev {r_t}: {e}")
                    continue

            try:
                proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                continue

            cal_y = np.array(calib_y_true[model_name])
            cal_p = np.array(calib_y_proba[model_name])
            if len(cal_y) >= 10 and len(np.unique(cal_y)) >= 2:
                threshold = tune_threshold(cal_y, cal_p)
            else:
                threshold = THRESHOLD_DEFAULT

            preds = (proba >= threshold).astype(int)

            for j in range(len(test)):
                row = test.iloc[j]
                all_predictions.append({
                    "model": model_name,
                    "verification_revision": int(r_t),
                    "gcid": int(row["gcid"]),
                    "classid": int(row["classid"]),
                    "decision_index": int(row["decision_index"]),
                    "prev_change_revision": int(row["prev_change_revision"]),
                    "y_true": int(y_test[j]),
                    "y_pred": int(preds[j]),
                    "y_proba": round(float(proba[j]), 4),
                    "threshold": round(threshold, 3),
                })

            rev_m = compute_metrics(y_test, preds, proba)
            rev_m["model"] = model_name
            rev_m["verification_revision"] = int(r_t)
            rev_m["threshold"] = round(threshold, 3)
            rev_m["train_size"] = len(train)
            per_rev_metrics.append(rev_m)

            global_y_true[model_name].extend(y_test.tolist())
            global_y_pred[model_name].extend(preds.tolist())
            global_y_proba[model_name].extend(proba.tolist())

            calib_y_true[model_name].extend(y_test.tolist())
            calib_y_proba[model_name].extend(proba.tolist())
            if len(calib_y_true[model_name]) > CALIBRATION_WINDOW_MAX:
                calib_y_true[model_name] = calib_y_true[model_name][-CALIBRATION_WINDOW_MAX:]
                calib_y_proba[model_name] = calib_y_proba[model_name][-CALIBRATION_WINDOW_MAX:]

        n_predicted += 1

        if n_predicted % 25 == 0:
            best_mcc = -2
            best_name = ""
            for mn in models:
                if len(global_y_true[mn]) > 0:
                    mcc = matthews_corrcoef(global_y_true[mn], global_y_pred[mn])
                    if mcc > best_mcc:
                        best_mcc = mcc
                        best_name = mn
            print(f"  Rev {r_t} ({n_predicted} predicted) | "
                  f"Best: {best_name} MCC={best_mcc:.3f} "
                  f"({len(global_y_true[best_name])} samples)")

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
        print(f"    Sensitivity  = {gm['sensitivity']:.4f}  (positive=dependent)")
        print(f"    Specificity  = {gm['specificity']:.4f}  (negative=independent)")
        print(f"    TP={gm['TP']}  TN={gm['TN']}  FP={gm['FP']}  FN={gm['FN']}")
        print(f"    Predictions  = {gm['total_predictions']}")

    # ── SAVE RESULTS ─────────────────────────────────────────────────────────
    global_df = pd.DataFrame(global_results)
    global_path = output_dir / f"classifier_metrics_{tag}.csv"
    global_df.to_csv(global_path, index=False)
    print(f"\n  Global metrics  -> {global_path}")

    rev_df = pd.DataFrame(per_rev_metrics)
    rev_path = output_dir / f"classifier_per_revision_{tag}.csv"
    rev_df.to_csv(rev_path, index=False)
    print(f"  Per-revision    -> {rev_path}")

    pred_df = pd.DataFrame(all_predictions)
    pred_path = output_dir / f"classifier_predictions_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions     -> {pred_path}")

    # ── ROLLING MCC PLOT ─────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 5))
        for model_name in models:
            model_rev = rev_df[rev_df["model"] == model_name].copy()
            if model_rev.empty:
                continue
            model_rev = model_rev.sort_values("verification_revision")
            model_rev["rolling_mcc"] = model_rev["mcc"].rolling(window=10, min_periods=1).mean()
            ax.plot(model_rev["verification_revision"], model_rev["rolling_mcc"],
                    label=model_name, alpha=0.8, linewidth=1.2)

        ax.set_xlabel("Verification Revision")
        ax.set_ylabel("Rolling MCC (window=10)")
        ax.set_title(f"Fragment Classifier: Rolling MCC Over Verification Revisions ({tag})")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plot_path = output_dir / f"classifier_mcc_over_revisions_{tag}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  MCC plot        -> {plot_path}")
    except Exception as e:
        print(f"  [WARN] Could not create plot: {e}")

    # ── FEATURE IMPORTANCE ───────────────────────────────────────────────────
    for model_name, model in last_fitted_models.items():
        try:
            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame({
                    "feature": feature_cols,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)
                imp_path = output_dir / f"classifier_feature_importance_{tag}_{model_name}.csv"
                imp.to_csv(imp_path, index=False)
                print(f"  Feature imp     -> {imp_path}")
        except Exception:
            pass

    # ── TOP-10 PRINT FOR BEST MODEL ──────────────────────────────────────────
    if global_results:
        best = max(global_results, key=lambda x: x["mcc"])
        best_name = best["model"]
        if best_name in last_fitted_models and hasattr(last_fitted_models[best_name], "feature_importances_"):
            imp = pd.DataFrame({
                "feature": feature_cols,
                "importance": last_fitted_models[best_name].feature_importances_,
            }).sort_values("importance", ascending=False)
            print(f"\n  Top 10 features ({best_name}, MCC={best['mcc']:.4f}):")
            for _, row in imp.head(10).iterrows():
                print(f"    {row['feature']:40s} {row['importance']:.4f}")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluator for the fragment-classifier dataset")
    parser.add_argument("--csv", required=True,
                        help="Path to {Type}_{Gran}_classifier_dataset.csv")
    parser.add_argument("--output", default=None,
                        help="Output directory for results (default: ml/results)")
    parser.add_argument("--min-warmup-revisions", type=int, default=MIN_WARMUP_REVISIONS,
                        help=f"Skip first N verification revisions (default: {MIN_WARMUP_REVISIONS})")
    parser.add_argument("--refit-every", type=int, default=REFIT_EVERY,
                        help=f"Retrain every N revisions (default: {REFIT_EVERY})")
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS,
                        help=f"Minimum training rows before predicting (default: {MIN_TRAIN_ROWS})")
    args = parser.parse_args()
    walk_forward(args.csv, args.output,
                 args.min_warmup_revisions, args.refit_every, args.min_train_rows)
