"""
predict_standing.py
────────────────────────────────────────────────────────────────────────────
Walk-forward clone-fragment change forecasting (sliding history).

Spec:
  Iterate revisions R where ≥1 fragment changes (the "validation revisions").
  At each such R, for every fragment g that changes at R:
    1. Build feature vector x(g, R) from layers 1+2+3 strictly up to R-1
    2. Predict ŷ(g) ∈ {dependent=1, independent=0}
    3. Compute actual y(g, R) = 1 if any peer of g also changed at R, else 0
    4. Log (R, g, x, ŷ, y) — feeds rolling metrics and periodic retraining

Inputs:
  WorkFolder/<subject>/Datasets/CloneGenealogy/<Type>_<Gran>_rev_fragment.csv
  WorkFolder/<subject>/Datasets/CloneGenealogy/<Type>_<Gran>_rev_pair.csv

Usage:
  python ml/predict_standing.py \\
    --frag WorkFolder/Ctags/Datasets/CloneGenealogy/Type3_Block_rev_fragment.csv \\
    --pair WorkFolder/Ctags/Datasets/CloneGenealogy/Type3_Block_rev_pair.csv
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, f1_score,
    precision_score, recall_score, balanced_accuracy_score,
)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import RandomForestClassifier


# ── CONFIG ───────────────────────────────────────────────────────────────────

WARMUP_REVS    = 5    # skip first N revisions before issuing predictions
MIN_TRAIN      = 30   # minimum verified labels before training first model
REFIT_EVERY    = 20   # retrain after every N new verified labels
CALIB_WINDOW   = 100  # last N verified events for threshold tuning
THRESHOLD_DEFAULT = 0.5
SENTINEL_AGE   = 9999  # placeholder for "never happened" (lastCoChangeAge=-1, etc.)

# Feature column order (must be stable across train/predict calls)
FRAG_NUMERIC = [
    "nlines", "totalChanges", "totalUnchanged",
    "stabilityIndex", "changeProneness", "lifespan",
    "activePairs", "meanWCS", "maxWCS", "minWCS",
    "stronglyCoupledPairs", "decoupledPairs",
]

# Per-fragment process/ownership values pulled side-aware from any pair row
FRAG_SIDE_COLS = ["churn", "distinctAuthors", "majorAuthorProp", "minorAuthorCount"]

PAIR_AGG_COLS = [
    "pair_count_alive",
    "max_co_change_count", "sum_co_change_count",
    "max_own_independent", "sum_own_independent",
    "mean_coupling_trend", "min_coupling_trend",
    "min_last_co_change_age", "min_last_solo_change_age",
    "mean_similarity", "max_similarity",
    "frac_same_file", "mean_depth",
]

# New pair-level aggregations from the enriched rev_pair CSV
PAIR_NEW_AGG_COLS = [
    "mean_wcs_recent", "max_wcs_recent", "min_wcs_recent",
    "max_solo_after_last_cochange",
    "frac_is_spcp", "class_size", "mean_class_size",
    "mean_same_author",
]

DECAY_HORIZONS = [10, 20, 30, 50, 75]
DECAY_AGG_COLS = (
    [f"mean_ir_decay_h{h}"   for h in DECAY_HORIZONS] +
    [f"mean_cs_decay_h{h}"   for h in DECAY_HORIZONS] +
    [f"mean_spcp_decay_h{h}" for h in DECAY_HORIZONS]
)

FEATURE_COLS = (
    FRAG_NUMERIC + FRAG_SIDE_COLS + PAIR_AGG_COLS + PAIR_NEW_AGG_COLS + DECAY_AGG_COLS
)


# ── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def aggregate_pair_features_for_revision(pair_df_at_rev, gcid):
    """Aggregate Layer-2 pair metrics for the given gcid at one revision.

    Returns a dict keyed by PAIR_AGG_COLS, or None if the gcid has no active pair
    at this revision (meaning its class has no co-existing peer)."""
    mask = (pair_df_at_rev["gcid1"] == gcid) | (pair_df_at_rev["gcid2"] == gcid)
    pairs = pair_df_at_rev[mask]
    if len(pairs) == 0:
        return None

    # "own_independent" depends on which side this gcid is on
    own_indep = np.where(
        pairs["gcid1"].values == gcid,
        pairs["gcid1Independent"].values,
        pairs["gcid2Independent"].values,
    )

    # Replace sentinel -1 ages with a large positive number so "min" is meaningful
    last_co  = pairs["lastCoChangeAge"].replace(-1, SENTINEL_AGE).values
    last_solo = pairs["lastSoloChangeAge"].replace(-1, SENTINEL_AGE).values

    out = {
        "pair_count_alive":        len(pairs),
        "max_co_change_count":     int(pairs["coChangeCount"].max()),
        "sum_co_change_count":     int(pairs["coChangeCount"].sum()),
        "max_own_independent":     int(own_indep.max()),
        "sum_own_independent":     int(own_indep.sum()),
        "mean_coupling_trend":     float(pairs["couplingTrend"].mean()),
        "min_coupling_trend":      float(pairs["couplingTrend"].min()),
        "min_last_co_change_age":  int(last_co.min()),
        "min_last_solo_change_age": int(last_solo.min()),
        "mean_similarity":         float(pairs["similarity"].mean()),
        "max_similarity":          float(pairs["similarity"].max()),
        "frac_same_file":          float(pairs["sameFile"].mean()),
        "mean_depth":              float(pairs["depth"].mean()),
    }

    # New pair-level aggregations (from enriched rev_pair.csv)
    if "wcsRecent" in pairs.columns:
        out["mean_wcs_recent"] = float(pairs["wcsRecent"].mean())
        out["max_wcs_recent"]  = float(pairs["wcsRecent"].max())
        out["min_wcs_recent"]  = float(pairs["wcsRecent"].min())
    else:
        out["mean_wcs_recent"] = out["max_wcs_recent"] = out["min_wcs_recent"] = 0.0

    out["max_solo_after_last_cochange"] = (
        int(pairs["soloEventsAfterLastCoChange"].max())
        if "soloEventsAfterLastCoChange" in pairs.columns else 0
    )
    out["frac_is_spcp"] = (
        float(pairs["isSpcp"].mean()) if "isSpcp" in pairs.columns else 0.0
    )
    if "classSize" in pairs.columns:
        out["class_size"]      = int(pairs["classSize"].iloc[0])
        out["mean_class_size"] = float(pairs["classSize"].mean())
    else:
        out["class_size"] = 1
        out["mean_class_size"] = 1.0
    out["mean_same_author"] = (
        float(pairs["sameAuthor"].mean()) if "sameAuthor" in pairs.columns else 0.0
    )

    # Decay-feature means (one mean per horizon per family)
    for h in DECAY_HORIZONS:
        for prefix, key in (("ir_decay", "mean_ir_decay"),
                            ("cs_decay", "mean_cs_decay"),
                            ("spcp_decay", "mean_spcp_decay")):
            col = f"{prefix}_h{h}"
            out[f"{key}_h{h}"] = float(pairs[col].mean()) if col in pairs.columns else 0.0

    return out


def extract_frag_side_features(pairs, gcid):
    """Pick churn / distinctAuthors / majorAuthorProp / minorAuthorCount for THIS gcid.

    These are per-fragment values that the Java exporter replicates on every pair
    row (as `_1` / `_2` columns). Pull side-1 if gcid is gcid1, else side-2.
    Returns zeros if no pairs are alive for this gcid."""
    if pairs is None or len(pairs) == 0:
        return {c: 0.0 for c in FRAG_SIDE_COLS}
    side1 = pairs[pairs["gcid1"] == gcid]
    if len(side1) > 0:
        r, suf = side1.iloc[0], "1"
    else:
        side2 = pairs[pairs["gcid2"] == gcid]
        if len(side2) == 0:
            return {c: 0.0 for c in FRAG_SIDE_COLS}
        r, suf = side2.iloc[0], "2"

    def col(name):
        return f"{name}{suf}"

    return {
        "churn":            float(r[col("churn")])            if col("churn") in r.index else 0.0,
        "distinctAuthors":  int(r[col("distinctAuthors")])    if col("distinctAuthors") in r.index else 0,
        "majorAuthorProp":  float(r[col("majorAuthorProp")])  if col("majorAuthorProp") in r.index else 0.0,
        "minorAuthorCount": int(r[col("minorAuthorCount")])   if col("minorAuthorCount") in r.index else 0,
    }


def _empty_pair_feats():
    """Zero-fill for fragments with no alive pairs at the lookup revision."""
    feats = {col: 0.0 for col in PAIR_AGG_COLS + PAIR_NEW_AGG_COLS + DECAY_AGG_COLS}
    feats["min_last_co_change_age"]   = SENTINEL_AGE
    feats["min_last_solo_change_age"] = SENTINEL_AGE
    feats["class_size"]      = 1
    feats["mean_class_size"] = 1.0
    return feats


def _last_rev_strictly_before(sorted_revs, R):
    """Return the largest revision in sorted_revs that is strictly < R, or None."""
    import bisect
    idx = bisect.bisect_left(sorted_revs, R)
    return sorted_revs[idx - 1] if idx > 0 else None


def build_feature_vector(gcid, R, frag_index, pair_index,
                         frag_revs_by_gcid, pair_revs_sorted):
    """Build the feature vector x(g, R) using strictly history < R.

    Both rev_fragment.csv and rev_pair.csv are sparse (M-events only), so we
    look up the most recent revision < R that has data for this gcid / for
    pairs in general — never R itself, to avoid label leakage.
    Returns numpy array of length len(FEATURE_COLS), or None if g has no
    prior history < R (cannot be predicted yet)."""
    # Fragment-level lookup: most recent M-row of THIS gcid before R
    gcid_revs = frag_revs_by_gcid.get(gcid)
    if not gcid_revs:
        return None
    frag_rev = _last_rev_strictly_before(gcid_revs, R)
    if frag_rev is None:
        return None
    frag_at = frag_index[frag_rev]
    row = frag_at[frag_at["gcid"] == gcid].iloc[0]

    # Pair-level lookup: most recent revision < R where any pair row exists
    pair_rev = _last_rev_strictly_before(pair_revs_sorted, R)
    pair_at = pair_index.get(pair_rev) if pair_rev is not None else None

    pair_feats = aggregate_pair_features_for_revision(pair_at, gcid) if pair_at is not None else None
    if pair_feats is None:
        pair_feats = _empty_pair_feats()

    side_feats = extract_frag_side_features(pair_at, gcid)

    vec = (
        [float(row[c]) for c in FRAG_NUMERIC]
        + [side_feats[c] for c in FRAG_SIDE_COLS]
        + [pair_feats[c] for c in PAIR_AGG_COLS]
        + [pair_feats[c] for c in PAIR_NEW_AGG_COLS]
        + [pair_feats[c] for c in DECAY_AGG_COLS]
    )
    return np.array(vec, dtype=float)


# ── LABEL COMPUTATION ───────────────────────────────────────────────────────

def compute_actual_label(gcid, rev, pair_index):
    """y(g, R') = 1 if any peer of g also has changeType=='M' at R', else 0."""
    pair_at = pair_index.get(rev)
    if pair_at is None:
        return 0
    mask1 = (pair_at["gcid1"] == gcid) & (pair_at["changeType2"].str.upper() == "M")
    mask2 = (pair_at["gcid2"] == gcid) & (pair_at["changeType1"].str.upper() == "M")
    return 1 if (mask1.any() or mask2.any()) else 0


# ── MODEL ────────────────────────────────────────────────────────────────────

def make_classifier():
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=10, is_unbalance=True,
            random_state=42, verbose=-1, n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


def tune_threshold(y_true, y_proba, n_points=50):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return THRESHOLD_DEFAULT
    best_mcc, best_t = -2, THRESHOLD_DEFAULT
    for t in np.linspace(0.1, 0.9, n_points):
        preds = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return best_t


# ── MAIN LOOP ────────────────────────────────────────────────────────────────

def run_standing_prediction(frag_path, pair_path, output_dir=None):
    print(f"Loading fragment CSV: {frag_path}")
    frag_df = pd.read_csv(frag_path)
    print(f"  Rows: {len(frag_df):,}, Revisions: {frag_df['revision'].nunique()}, "
          f"Unique gcids: {frag_df['gcid'].nunique()}")

    print(f"Loading pair CSV:     {pair_path}")
    pair_df = pd.read_csv(pair_path)
    print(f"  Rows: {len(pair_df):,}")

    # Drop gcids that never underwent any 'M' event (totalChanges stays 0 forever).
    # These carry no label signal and should not enter training or prediction.
    ever_changed = frag_df.groupby("gcid")["totalChanges"].max()
    active_gcids = set(ever_changed[ever_changed > 0].index)
    before_frag = len(frag_df)
    frag_df = frag_df[frag_df["gcid"].isin(active_gcids)].copy()
    print(f"  Fragment filter: {frag_df['gcid'].nunique()} gcids retained "
          f"({before_frag - len(frag_df):,} rows removed, totalChanges-always-0 gcids dropped)")

    # Drop pair rows where both gcids never changed (no signal for either side).
    before_pair = len(pair_df)
    pair_df = pair_df[
        pair_df["gcid1"].isin(active_gcids) | pair_df["gcid2"].isin(active_gcids)
    ].copy()
    print(f"  Pair filter:     {before_pair - len(pair_df):,} rows removed\n")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_stem = Path(frag_path).stem.replace("_rev_fragment", "")
    subject = Path(frag_path).parent.parent.parent.name
    tag = f"{subject}_{csv_stem}"
    print(f"  Tag: {tag}\n")

    # Pre-index by revision for O(1) lookups
    print("Indexing by revision...")
    frag_index = {r: g for r, g in frag_df.groupby("revision")}
    pair_index = {r: g for r, g in pair_df.groupby("revision")}

    # Sparse-CSV lookups: rev_fragment & rev_pair only contain rows at
    # change revisions, so a strict R-1 lookup misses fragments that didn't
    # change at R-1. Pre-build sorted revision lists per gcid (and globally
    # for pairs) to support "most recent revision strictly < R" queries.
    frag_revs_by_gcid = {
        g: sorted(grp["revision"].tolist())
        for g, grp in frag_df.groupby("gcid")
    }
    pair_revs_sorted = sorted(pair_df["revision"].unique().tolist())

    # Iterate ONLY revisions where ≥1 fragment changed — these are the
    # "validation revisions" in the user's spec. At each such R we predict and
    # verify the fragments changing at R, using history strictly < R.
    change_revisions = sorted(
        frag_df.loc[frag_df["changeType"].str.upper() == "M", "revision"].unique()
    )
    print(f"Walking {len(change_revisions)} change revisions "
          f"(warmup = {WARMUP_REVS} revs, MIN_TRAIN = {MIN_TRAIN})\n")

    # State
    verification_log = []  # one row per change event (features + pred + actual)
    metrics_over_time = []

    model = None
    threshold = THRESHOLD_DEFAULT
    last_refit_count = 0

    for idx, R in enumerate(change_revisions):
        alive_at_R = frag_index.get(R)
        if alive_at_R is None or len(alive_at_R) == 0:
            continue
        changed = alive_at_R[alive_at_R["changeType"].str.upper() == "M"]

        # ── 1. Predict + verify each fragment changing at R ──────────────────
        for _, row in changed.iterrows():
            g = int(row["gcid"])
            x_vec = build_feature_vector(
                g, R, frag_index, pair_index,
                frag_revs_by_gcid, pair_revs_sorted,
            )
            if x_vec is None:
                continue
            actual_y = compute_actual_label(g, R, pair_index)

            pred_label, pred_proba, correct = None, None, None
            if model is not None and idx >= WARMUP_REVS:
                try:
                    pred_proba = float(model.predict_proba([x_vec])[0][1])
                    pred_label = int(pred_proba >= threshold)
                    correct    = int(pred_label == actual_y)
                except Exception:
                    pred_label = pred_proba = correct = None

            entry = {
                "revision":     R,
                "gcid":         g,
                "actual_label": actual_y,
                "pred_label":   pred_label,
                "pred_proba":   None if pred_proba is None else round(pred_proba, 4),
                "correct":      correct,
                "threshold":    round(threshold, 3),
            }
            for col, val in zip(FEATURE_COLS, x_vec):
                entry[col] = val
            verification_log.append(entry)

        # ── 2. Retrain trigger ───────────────────────────────────────────────
        # Every logged row carries an actual label and full feature vector,
        # so the labeled set is just the verification log itself.
        n_labeled = len(verification_log)

        if n_labeled >= MIN_TRAIN and (n_labeled - last_refit_count >= REFIT_EVERY
                                      or model is None):
            X = np.array([[r[c] for c in FEATURE_COLS] for r in verification_log])
            y = np.array([r["actual_label"] for r in verification_log])
            if len(np.unique(y)) >= 2:
                model = make_classifier()
                model.fit(X, y)
                last_refit_count = n_labeled

                # Re-tune threshold on the most recent CALIB_WINDOW labels
                recent = verification_log[-CALIB_WINDOW:]
                Xc = np.array([[r[c] for c in FEATURE_COLS] for r in recent])
                yc = np.array([r["actual_label"] for r in recent])
                try:
                    pc = model.predict_proba(Xc)[:, 1]
                    threshold = tune_threshold(yc, pc)
                except Exception:
                    threshold = THRESHOLD_DEFAULT

                print(f"  [retrain] rev={R} n_labeled={n_labeled} "
                      f"threshold={threshold:.3f}")

        # ── 3. Rolling metrics over verified-with-prediction events ──────────
        verified_with_pred = [r for r in verification_log if r["correct"] is not None]
        if verified_with_pred:
            yt = np.array([r["actual_label"] for r in verified_with_pred])
            yp = np.array([r["pred_label"]   for r in verified_with_pred])
            metrics_over_time.append({
                "revision":                R,
                "n_change_events_so_far":  len(verification_log),
                "n_verified_so_far":       len(verified_with_pred),
                "rolling_acc":             round(accuracy_score(yt, yp), 4),
                "rolling_mcc":             round(matthews_corrcoef(yt, yp), 4)
                                            if len(np.unique(yt)) >= 2 else 0.0,
                "rolling_f1":              round(f1_score(yt, yp, zero_division=0), 4),
                "threshold":               round(threshold, 3),
            })

        if (idx + 1) % 25 == 0:
            verified = sum(1 for r in verification_log if r["correct"] is not None)
            print(f"  rev {R} ({idx+1}/{len(change_revisions)}) "
                  f"events={len(verification_log)} verified={verified}")

    # ── GLOBAL SUMMARY ───────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  GLOBAL SUMMARY: {tag}\n{'='*70}")
    verified_only = [r for r in verification_log if r["correct"] is not None]
    if verified_only:
        yt = np.array([r["actual_label"] for r in verified_only])
        yp = np.array([r["pred_label"]   for r in verified_only])
        print(f"  Total verifications:  {len(verification_log):,}")
        print(f"  Of which had pred:    {len(verified_only):,}")
        print(f"  Accuracy:             {accuracy_score(yt, yp):.4f}")
        if len(np.unique(yt)) >= 2:
            print(f"  MCC:                  {matthews_corrcoef(yt, yp):.4f}")
            print(f"  Balanced accuracy:    {balanced_accuracy_score(yt, yp):.4f}")
        print(f"  F1 (label=1=dep):     {f1_score(yt, yp, zero_division=0):.4f}")
        print(f"  Precision (dep):      {precision_score(yt, yp, zero_division=0):.4f}")
        print(f"  Recall (dep):         {recall_score(yt, yp, zero_division=0):.4f}")
    else:
        print("  No verified predictions (warmup never ended).")

    # ── WRITE OUTPUTS ────────────────────────────────────────────────────────
    verify_path  = output_dir / f"standing_{tag}_verification_log.csv"
    metrics_path = output_dir / f"standing_{tag}_metrics_over_time.csv"

    pd.DataFrame(verification_log).to_csv(verify_path, index=False)
    print(f"\n  Verification log  -> {verify_path}")

    pd.DataFrame(metrics_over_time).to_csv(metrics_path, index=False)
    print(f"  Metrics over time -> {metrics_path}")

    # Persist final model
    if model is not None:
        try:
            import joblib
            model_path = output_dir / f"standing_{tag}_final_model.pkl"
            joblib.dump({"model": model, "features": FEATURE_COLS,
                         "threshold": threshold}, model_path)
            print(f"  Final model       -> {model_path}")
        except Exception as e:
            print(f"  [WARN] Could not save model: {e}")

    # Rolling MCC plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if metrics_over_time:
            mdf = pd.DataFrame(metrics_over_time)
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(mdf["revision"], mdf["rolling_mcc"], label="Rolling MCC",
                    linewidth=1.4)
            ax.plot(mdf["revision"], mdf["rolling_acc"], label="Rolling Accuracy",
                    linewidth=1.0, alpha=0.7)
            ax.plot(mdf["revision"], mdf["rolling_f1"],  label="Rolling F1",
                    linewidth=1.0, alpha=0.7)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Revision")
            ax.set_ylabel("Score")
            ax.set_title(f"Standing Prediction Performance Over Revisions ({tag})")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            plot_path = output_dir / f"standing_{tag}_mcc_over_revisions.png"
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"  MCC plot          -> {plot_path}")
    except Exception as e:
        print(f"  [WARN] Could not create plot: {e}")

    print(f"\n{'='*70}\n  DONE\n{'='*70}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standing-prediction + verification loop for clone fragments")
    parser.add_argument("--frag", required=True,
                        help="Path to <Type>_<Gran>_rev_fragment.csv")
    parser.add_argument("--pair", required=True,
                        help="Path to <Type>_<Gran>_rev_pair.csv")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: ml/results)")
    args = parser.parse_args()
    run_standing_prediction(args.frag, args.pair, args.output)
