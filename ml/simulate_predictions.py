"""
simulate_predictions.py  —  Full step-by-step walk-forward simulation
======================================================================
Shows EVERY step of the pipeline for the tuxguitar dataset:
  • Warmup: feature building sources, label derivation, pool growth
  • Training trigger: class balance, model fit
  • Prediction: raw proba → threshold → binary decision
  • Verification: ground-truth source, confusion matrix, precision formula
  • Retraining & threshold tuning after each round

Usage:
    py simulate_predictions.py [--max-preds 6]

predict_standing.py is NOT modified.
"""
import sys, os, bisect, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_standing as ps

_BASE     = Path(__file__).resolve().parent.parent / "WorkFolder" / "tuxguitar" / "Datasets" / "CloneGenealogy"
FRAG_PATH = str(_BASE / "Type3_Block_rev_fragment.csv")
PAIR_PATH = str(_BASE / "Type3_Block_rev_pair.csv")

MAX_PRED_ROUNDS = 6
W = 72          # banner width

KEY_FRAG_FEATS  = ["nlines", "totalChanges", "stabilityIndex", "changeProneness", "lifespan"]
KEY_PAIR_FEATS  = ["pair_count_alive", "max_co_change_count", "mean_coupling_trend",
                   "mean_similarity", "frac_same_file"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _banner(title, char="="):
    pad = max(0, W - len(title) - 4)
    print(f"\n{char*2} {title} {char * pad}")


def _sub(title):
    print(f"\n  >> {title}")
    print(f"  {'─'*66}")


def _show_fv(gcid, xv, frag_source_rev, pair_source_rev):
    """Print key feature values for one fragment's feature vector."""
    feat_names = ps.FRAG_NUMERIC + ps.FRAG_SIDE_COLS + ps.PAIR_AGG_COLS + ps.PAIR_ENRICHED + ps.DECAY_AGG
    fv_dict    = dict(zip(feat_names, xv))
    frag_vals  = {k: fv_dict.get(k, 0) for k in KEY_FRAG_FEATS}
    pair_vals  = {k: fv_dict.get(k, 0) for k in KEY_PAIR_FEATS}
    print(f"     gcid={gcid}  [SOURCE A=rev_fragment@{frag_source_rev}, SOURCE B=rev_pair@{pair_source_rev}]")
    frag_str = "  ".join(f"{k}={v:.3g}" for k, v in frag_vals.items())
    pair_str = "  ".join(f"{k}={v:.3g}" for k, v in pair_vals.items())
    print(f"       Frag: {frag_str}")
    print(f"       Pair: {pair_str}")
    print(f"       (total dims={len(xv)})")


def _show_label_derivation(gcid, R, pair_index):
    """Explain exactly how actual_label is derived from rev_pair AT revision R."""
    pat = pair_index.get(R)
    if pat is None:
        print(f"     gcid={gcid}: no pair row at rev {R} → label=0 (INDEPENDENT)")
        return 0
    m1 = (pat["gcid1"] == gcid) & (pat["changeType2"].str.upper() == "M")
    m2 = (pat["gcid2"] == gcid) & (pat["changeType1"].str.upper() == "M")
    label = 1 if (m1.any() or m2.any()) else 0
    peer_rows = pat[m1 | m2]
    if label == 1:
        peers = list(peer_rows["gcid2"].where(pat["gcid1"] == gcid)
                     .dropna().astype(int))
        peers += list(peer_rows["gcid1"].where(pat["gcid2"] == gcid)
                      .dropna().astype(int))
        print(f"     gcid={gcid}: peer(s) {peers} also have changeType='M' at rev {R}")
        print(f"       → label = 1  (DEPENDENT)")
    else:
        print(f"     gcid={gcid}: no peer has changeType='M' at rev {R}")
        print(f"       → label = 0  (INDEPENDENT)")
    return label


def _show_prediction(mn, gcid, xv, model, threshold):
    """Show raw proba and threshold decision for one fragment."""
    proba = model.predict_proba(xv.reshape(1, -1))[0, 1]
    pred  = int(proba >= threshold)
    decision = "DEPENDENT" if pred == 1 else "INDEPENDENT"
    sign = ">=" if pred == 1 else " <"
    print(f"     gcid={gcid}: proba={proba:.4f}  {sign} threshold={threshold:.3f}  "
          f"→ pred={decision}")
    return pred, proba


def _show_confusion_and_metrics(gcids, labels, preds, probas, threshold):
    """Print full confusion matrix and compute precision step-by-step."""
    tp = tn = fp = fn = 0
    for lbl, prd in zip(labels, preds):
        if   prd == 1 and lbl == 1: tp += 1
        elif prd == 0 and lbl == 0: tn += 1
        elif prd == 1 and lbl == 0: fp += 1
        else:                       fn += 1

    print(f"     Confusion Matrix (threshold={threshold:.3f}):")
    print(f"                        Pred DEPENDENT  Pred INDEPENDENT")
    print(f"       Actual DEPENDENT       TP={tp:<4}         FN={fn}")
    print(f"       Actual INDEPENDENT     FP={fp:<4}         TN={tn}")
    print()

    n    = len(labels)
    acc  = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    yt   = np.array(labels); yp = np.array(preds)
    mcc  = ps.matthews_corrcoef(yt, yp) if len(np.unique(yt)) >= 2 else 0.0

    print(f"     Metric formulas:")
    print(f"       Precision  = TP/(TP+FP)  = {tp}/({tp}+{fp}) = {prec:.4f}")
    print(f"       Recall     = TP/(TP+FN)  = {tp}/({tp}+{fn}) = {rec:.4f}")
    print(f"       Specificity= TN/(TN+FP)  = {tn}/({tn}+{fp}) = {spec:.4f}")
    print(f"       Accuracy   = (TP+TN)/n   = ({tp}+{tn})/{n}  = {acc:.4f}")
    print(f"       MCC        = {mcc:.4f}  (Matthews Correlation Coefficient)")
    return tp, tn, fp, fn


# ── main simulation ──────────────────────────────────────────────────────────

def simulate(max_pred_rounds=MAX_PRED_ROUNDS):

    _banner(f"FULL STEP-BY-STEP SIMULATION  —  tuxguitar / Type3_Block")
    print(f"  Post-warmup prediction rounds to show : {max_pred_rounds}")
    print(f"  Warmup revisions (no prediction)      : {ps.WARMUP_REVS}")
    print(f"  Min training samples before first fit : {ps.MIN_TRAIN}")
    print(f"  Calibration window (threshold tuning) : {ps.CALIB_WINDOW}")

    # ── STEP 0: Load data ────────────────────────────────────────────────────
    _banner("STEP 0 — Load & Index Data", "-")
    print(f"  Loading: {FRAG_PATH}")
    frag_df = pd.read_csv(FRAG_PATH)
    print(f"  rev_fragment: {len(frag_df):,} rows  "
          f"| {frag_df['revision'].nunique()} revisions  "
          f"| {frag_df['gcid'].nunique()} gcids")

    print(f"  Loading: {PAIR_PATH}")
    pair_df = pd.read_csv(PAIR_PATH)
    print(f"  rev_pair:     {len(pair_df):,} rows  | {len(pair_df.columns)} columns")

    ever   = frag_df.groupby("gcid")["totalChanges"].max()
    active = set(ever[ever > 0].index)
    frag_df = frag_df[frag_df["gcid"].isin(active)].copy()
    pair_df = pair_df[pair_df["gcid1"].isin(active) | pair_df["gcid2"].isin(active)].copy()
    print(f"  After filtering inactive gcids: "
          f"{len(frag_df):,} fragment rows, {len(pair_df):,} pair rows")

    frag_index = {r: g for r, g in frag_df.groupby("revision")}
    pair_index = {r: g for r, g in pair_df.groupby("revision")}
    frag_revs  = {g: sorted(grp["revision"].tolist()) for g, grp in frag_df.groupby("gcid")}
    pair_revs  = sorted(pair_df["revision"].unique().tolist())
    change_revisions = sorted(
        frag_df.loc[frag_df["changeType"].str.upper() == "M", "revision"].unique())
    print(f"  Change revisions : {len(change_revisions)}")
    print(f"  Pair revisions   : {len(pair_revs)}")
    print(f"  Feature vector   : {len(ps.FEATURE_COLS)} dimensions")
    print(f"    Layer 1 (fragment): {len(ps.FRAG_NUMERIC)+len(ps.FRAG_SIDE_COLS)} dims")
    print(f"    Layer 2 (pair agg): {len(ps.PAIR_AGG_COLS)+len(ps.PAIR_ENRICHED)} dims")
    print(f"    Decay features   : {len(ps.DECAY_AGG)} dims  (horizons={ps.DECAY_HORIZONS})")

    models      = ps.get_models()
    model_names = list(models.keys())
    print(f"  Models: {model_names}")

    thresholds = {mn: ps.THRESHOLD_DEFAULT for mn in model_names}
    fitted     = {mn: False               for mn in model_names}
    last_refit = {mn: 0                   for mn in model_names}
    calib      = {mn: {"yt": [], "yp": []} for mn in model_names}
    all_X, all_y = [], []
    pred_round_count = 0

    train_rev_count = 0   # post-warmup revisions before first prediction

    for idx, R in enumerate(change_revisions):
        alive = frag_index.get(R)
        if alive is None:
            continue
        changed = alive[alive["changeType"].str.upper() == "M"]
        is_warmup  = (idx < ps.WARMUP_REVS)
        all_fitted = all(fitted.values())

        # Informative phase label: WARMUP → POOL-BUILDING → PREDICTION ROUND
        if is_warmup:
            phase = f"WARMUP {idx+1}/{ps.WARMUP_REVS}"
            hchar = "-"
        elif not all_fitted:
            train_rev_count += 1
            phase = f"POOL-BUILDING rev {train_rev_count}  (pre-train, models not fitted yet)"
            hchar = "-"
        else:
            phase = f"PREDICTION ROUND {pred_round_count+1}"
            hchar = "="

        _banner(f"{phase}  —  rev={R}", hchar)

        changed_gcids = sorted(changed["gcid"].tolist())
        print(f"  Changed gcids at rev {R}: {changed_gcids}  ({len(changed_gcids)} total)")

        # ── SUB-STEP A: Build Feature Vectors ─────────────────────────────
        _sub("SUB-STEP A: Build Feature Vectors  (data strictly BEFORE rev " + str(R) + ")")
        print(f"  Rule: last fragment snapshot at rev < {R}  +  last pair snapshot at rev < {R}")
        rev_vecs, rev_labels, rev_gcids = [], [], []
        skipped = 0
        pidx_global = bisect.bisect_left(pair_revs, R)
        prev_global = pair_revs[pidx_global-1] if pidx_global > 0 else None

        for _, row in changed.iterrows():
            g     = int(row["gcid"])
            grevs = frag_revs.get(g, [])
            fidx  = bisect.bisect_left(grevs, R)
            frev  = grevs[fidx-1] if fidx > 0 else None
            xv    = ps.build_fv(g, R, frag_index, pair_index, frag_revs, pair_revs)
            if xv is None:
                skipped += 1
                continue
            _show_fv(g, xv, frev, prev_global)

            # ── SUB-STEP B: Label derivation ────────────────────────────
            _sub("SUB-STEP B: Derive Actual Label  (data AT rev " + str(R) + ")")
            print(f"  Rule: label=1 iff ∃ peer p: pair(gcid,p) AND changeType(p,R)='M'")
            ay = _show_label_derivation(g, R, pair_index)
            rev_vecs.append(xv); rev_labels.append(ay); rev_gcids.append(g)

        if skipped:
            print(f"\n     [{skipped} gcid(s) had no prior snapshot before rev {R} → skipped]")
        if not rev_vecs:
            print("  No valid feature vectors this revision — nothing to predict/train.")
            continue

        yr     = np.array(rev_labels)
        n_dep_r = int(yr.sum())
        print(f"\n  Valid vectors this rev: {len(rev_gcids)}  "
              f"(DEP={n_dep_r}, IND={len(rev_gcids)-n_dep_r})")

        any_prediction = False

        # ── SUB-STEP C/D/E: Predict + Verify + Metrics ────────────────────
        if not is_warmup:
            _sub(f"SUB-STEP C: Predict with Each Model")
            for mn in model_names:
                print(f"\n     [{mn}]  threshold={thresholds[mn]:.3f}  fitted={fitted[mn]}")
                if not fitted[mn]:
                    print(f"       → Not yet fitted. Skipping prediction this round.")
                    continue

                preds_list, probas_list = [], []
                for g, xv in zip(rev_gcids, rev_vecs):
                    pred, proba = _show_prediction(mn, g, xv, models[mn], thresholds[mn])
                    preds_list.append(pred); probas_list.append(proba)

                preds  = np.array(preds_list)
                probas = np.array(probas_list)

                _sub(f"SUB-STEP D: Verify Predictions vs Ground Truth  [{mn}]")
                print(f"     {'gcid':>6}  {'Actual':>12}  {'Predicted':>12}  "
                      f"{'Proba':>8}  {'Verdict':>8}")
                print(f"     {'─'*6}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")
                for j in range(len(rev_gcids)):
                    lbl = "DEPENDENT"   if rev_labels[j] == 1 else "INDEPENDENT"
                    prd = "DEPENDENT"   if preds[j]      == 1 else "INDEPENDENT"
                    ok  = "CORRECT" if preds[j] == rev_labels[j] else "WRONG"
                    print(f"     {rev_gcids[j]:>6}  {lbl:>12}  {prd:>12}  "
                          f"  {probas[j]:.4f}  {ok}")

                _sub(f"SUB-STEP E: Compute Metrics  [{mn}]")
                _show_confusion_and_metrics(rev_gcids, rev_labels,
                                            preds.tolist(), probas.tolist(),
                                            thresholds[mn])

                calib[mn]["yt"].extend(yr.tolist())
                calib[mn]["yp"].extend(probas.tolist())
                if len(calib[mn]["yt"]) > ps.CALIB_WINDOW:
                    calib[mn]["yt"] = calib[mn]["yt"][-ps.CALIB_WINDOW:]
                    calib[mn]["yp"] = calib[mn]["yp"][-ps.CALIB_WINDOW:]
                any_prediction = True

        # ── SUB-STEP F: Update pool & retrain ─────────────────────────────
        all_X.extend(rev_vecs); all_y.extend(rev_labels)
        n_lab = len(all_y)
        n_dep = int(np.sum(all_y))
        n_ind = n_lab - n_dep

        _sub("SUB-STEP F: Update Training Pool & Retrain")
        print(f"     Pool after rev {R}: {n_lab} samples  (DEP={n_dep}, IND={n_ind})")

        can_train = (n_lab >= ps.MIN_TRAIN) and (len(np.unique(all_y)) >= 2)
        if not can_train:
            reason = (f"need {ps.MIN_TRAIN - n_lab} more to reach MIN_TRAIN={ps.MIN_TRAIN}"
                      if n_lab < ps.MIN_TRAIN else "only one class in pool")
            print(f"     ✗ Training NOT triggered: {reason}")
        else:
            Xa, ya = np.array(all_X), np.array(all_y)
            for mn in model_names:
                if n_lab - last_refit[mn] >= ps.REFIT_EVERY or not fitted[mn]:
                    trigger = "first fit" if not fitted[mn] else f"+{n_lab-last_refit[mn]} new"
                    print(f"     ✓ [{mn}] Fitting on {n_lab} samples ({trigger})  "
                          f"DEP={n_dep}, IND={n_ind}")
                    try:
                        m = models[mn]
                        if mn == "XGBoost":
                            spw = n_ind / n_dep if n_dep > 0 else 1.0
                            m.set_params(scale_pos_weight=spw)
                            print(f"         XGBoost scale_pos_weight = {n_ind}/{n_dep} = {spw:.2f}")
                        m.fit(Xa, ya)
                        fitted[mn] = True; last_refit[mn] = n_lab
                        cy = np.array(calib[mn]["yt"][-ps.CALIB_WINDOW:])
                        cp = np.array(calib[mn]["yp"][-ps.CALIB_WINDOW:])
                        if len(cy) >= 10 and len(np.unique(cy)) >= 2:
                            old_t = thresholds[mn]
                            thresholds[mn] = ps.tune_threshold(cy, cp)
                            print(f"         Threshold: {old_t:.3f} → {thresholds[mn]:.3f}  "
                                  f"(MCC-opt on {len(cy)} calib samples)")
                        else:
                            print(f"         Threshold: kept {thresholds[mn]:.3f}  "
                                  f"(calib={len(cy)} samples, need ≥10 both classes)")
                    except Exception as e:
                        print(f"         [WARN] fit failed: {e}")
                else:
                    print(f"     - [{mn}] No refit  "
                          f"({n_lab-last_refit[mn]} new, REFIT_EVERY={ps.REFIT_EVERY})")

        if any_prediction:
            pred_round_count += 1
            if pred_round_count >= max_pred_rounds:
                _banner(f"SIMULATION COMPLETE — {max_pred_rounds} prediction rounds shown")
                return


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Full step-by-step walk-forward simulation for tuxguitar")
    p.add_argument("--max-preds", type=int, default=MAX_PRED_ROUNDS,
                   help=f"Number of post-warmup prediction rounds (default {MAX_PRED_ROUNDS})")
    a = p.parse_args()
    simulate(max_pred_rounds=a.max_preds)
