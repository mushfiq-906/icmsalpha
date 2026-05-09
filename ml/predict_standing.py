"""
predict_standing.py — Walk-forward clone-fragment change forecasting.
Labels: 1=dependent (peer co-changed), 0=independent (changed alone).
Uses rev_fragment.csv (Layer 1+3) and rev_pair.csv (Layer 2).
"""
import argparse, bisect, os, warnings, time
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score,
)
try:
    from lightgbm import LGBMClassifier; HAS_LGBM = True
except ImportError: HAS_LGBM = False
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError: HAS_XGB = False
try:
    from catboost import CatBoostClassifier; HAS_CAT = True
except ImportError: HAS_CAT = False

# ── CONFIG ──
WARMUP_REVS = 5
MIN_TRAIN = 30
REFIT_EVERY = 1
CALIB_WINDOW = 50
THRESHOLD_DEFAULT = 0.5
SENTINEL_AGE = 9999
DECAY_HORIZONS = [10, 20, 30, 50, 75]

# Feature columns — Layer 1 (fragment)
FRAG_NUMERIC = [
    "nlines","totalChanges","totalUnchanged","stabilityIndex",
    "changeProneness","lifespan","activePairs","meanWCS","maxWCS",
    "minWCS","stronglyCoupledPairs","decoupledPairs",
]
FRAG_SIDE_COLS = ["churn","distinctAuthors","majorAuthorProp","minorAuthorCount"]

# Pair aggregation columns
PAIR_AGG_COLS = [
    "pair_count_alive","max_co_change_count","sum_co_change_count",
    "max_own_independent","sum_own_independent",
    "mean_coupling_trend","min_coupling_trend",
    "min_last_co_change_age","min_last_solo_change_age",
    "mean_similarity","max_similarity","frac_same_file","mean_depth",
]
PAIR_ENRICHED = [
    "mean_wcs_recent","max_wcs_recent","min_wcs_recent",
    "max_solo_after_last_cochange","frac_is_spcp",
    "class_size","mean_class_size","mean_same_author",
]
DECAY_AGG = (
    [f"mean_ir_decay_h{h}" for h in DECAY_HORIZONS] +
    [f"mean_cs_decay_h{h}" for h in DECAY_HORIZONS] +
    [f"mean_spcp_decay_h{h}" for h in DECAY_HORIZONS]
)
FEATURE_COLS = FRAG_NUMERIC + FRAG_SIDE_COLS + PAIR_AGG_COLS + PAIR_ENRICHED + DECAY_AGG


def _safe(df, col, default=0.0):
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


def aggregate_pairs(pair_at, gcid):
    mask = (pair_at["gcid1"] == gcid) | (pair_at["gcid2"] == gcid)
    pairs = pair_at[mask]
    if len(pairs) == 0:
        return None
    own_indep = np.where(pairs["gcid1"].values == gcid,
                         pairs["gcid1Independent"].values,
                         pairs["gcid2Independent"].values)
    last_co = pairs["lastCoChangeAge"].replace(-1, SENTINEL_AGE).values
    last_solo = pairs["lastSoloChangeAge"].replace(-1, SENTINEL_AGE).values
    o = {
        "pair_count_alive": len(pairs),
        "max_co_change_count": int(pairs["coChangeCount"].max()),
        "sum_co_change_count": int(pairs["coChangeCount"].sum()),
        "max_own_independent": int(own_indep.max()),
        "sum_own_independent": int(own_indep.sum()),
        "mean_coupling_trend": float(pairs["couplingTrend"].mean()),
        "min_coupling_trend": float(pairs["couplingTrend"].min()),
        "min_last_co_change_age": int(last_co.min()),
        "min_last_solo_change_age": int(last_solo.min()),
        "mean_similarity": float(pairs["similarity"].mean()),
        "max_similarity": float(pairs["similarity"].max()),
        "frac_same_file": float(pairs["sameFile"].mean()),
        "mean_depth": float(pairs["depth"].mean()),
    }
    o["mean_wcs_recent"] = float(_safe(pairs,"wcsRecent",0).mean())
    o["max_wcs_recent"] = float(_safe(pairs,"wcsRecent",0).max())
    o["min_wcs_recent"] = float(_safe(pairs,"wcsRecent",0).min())
    o["max_solo_after_last_cochange"] = int(_safe(pairs,"soloEventsAfterLastCoChange",0).max())
    o["frac_is_spcp"] = float(_safe(pairs,"isSpcp",0).mean())
    cs = _safe(pairs,"classSize",1)
    o["class_size"] = int(cs.iloc[0]); o["mean_class_size"] = float(cs.mean())
    o["mean_same_author"] = float(_safe(pairs,"sameAuthor",0).mean())
    for h in DECAY_HORIZONS:
        for pfx, key in [("ir_decay","mean_ir_decay"),("cs_decay","mean_cs_decay"),("spcp_decay","mean_spcp_decay")]:
            col = f"{pfx}_h{h}"
            o[f"{key}_h{h}"] = float(_safe(pairs,col,0).mean())
    return o


def extract_side_feats(pairs, gcid):
    if pairs is None or len(pairs) == 0:
        return {c: 0.0 for c in FRAG_SIDE_COLS}
    s1 = pairs[pairs["gcid1"] == gcid]
    if len(s1) > 0: r, suf = s1.iloc[0], "1"
    else:
        s2 = pairs[pairs["gcid2"] == gcid]
        if len(s2) == 0: return {c: 0.0 for c in FRAG_SIDE_COLS}
        r, suf = s2.iloc[0], "2"
    out = {}
    for c in FRAG_SIDE_COLS:
        cn = f"{c}{suf}"
        out[c] = float(r[cn]) if cn in r.index else 0.0
    return out


def empty_pair_feats():
    f = {c: 0.0 for c in PAIR_AGG_COLS + PAIR_ENRICHED + DECAY_AGG}
    f["min_last_co_change_age"] = SENTINEL_AGE
    f["min_last_solo_change_age"] = SENTINEL_AGE
    f["class_size"] = 1; f["mean_class_size"] = 1.0
    return f


def build_fv(gcid, R, frag_index, pair_index, frag_revs, pair_revs):
    grevs = frag_revs.get(gcid)
    if not grevs: return None
    idx = bisect.bisect_left(grevs, R)
    frev = grevs[idx-1] if idx > 0 else None
    if frev is None: return None
    row = frag_index[frev]
    row = row[row["gcid"] == gcid].iloc[0]
    pidx = bisect.bisect_left(pair_revs, R)
    prev = pair_revs[pidx-1] if pidx > 0 else None
    pat = pair_index.get(prev) if prev else None
    pf = aggregate_pairs(pat, gcid) if pat is not None else None
    if pf is None: pf = empty_pair_feats()
    sf = extract_side_feats(pat, gcid)
    vec = ([float(row[c]) for c in FRAG_NUMERIC]
           + [sf[c] for c in FRAG_SIDE_COLS]
           + [pf[c] for c in PAIR_AGG_COLS]
           + [pf[c] for c in PAIR_ENRICHED]
           + [pf[c] for c in DECAY_AGG])
    return np.array(vec, dtype=float)


def actual_label(gcid, rev, pair_index):
    pat = pair_index.get(rev)
    if pat is None: return 0
    m1 = (pat["gcid1"]==gcid) & (pat["changeType2"].str.upper()=="M")
    m2 = (pat["gcid2"]==gcid) & (pat["changeType1"].str.upper()=="M")
    return 1 if (m1.any() or m2.any()) else 0


def get_models():
    m = {}
    m["RandomForest"] = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1)
    if HAS_LGBM:
        m["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=10, is_unbalance=True,
            random_state=42, verbose=-1, n_jobs=-1)
    if HAS_XGB:
        m["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_weight=5, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1)
    if HAS_CAT:
        m["CatBoost"] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=42, verbose=0)
    return m


def compute_metrics(yt, yp, ypr=None):
    m = {"n": len(yt), "n_pos": int(yt.sum()), "n_neg": int((yt==0).sum())}
    if len(np.unique(yt)) < 2:
        m.update(mcc=0, auc_roc=0.5, pr_auc=0, balanced_acc=0.5, f1=0, gmean=0,
                 sensitivity=0, specificity=0, TP=0, TN=0, FP=0, FN=0)
        return m
    m["mcc"] = matthews_corrcoef(yt, yp)
    m["balanced_acc"] = balanced_accuracy_score(yt, yp)
    m["f1"] = f1_score(yt, yp, zero_division=0)
    if ypr is not None:
        try: m["auc_roc"] = roc_auc_score(yt, ypr)
        except: m["auc_roc"] = 0.5
        try: m["pr_auc"] = average_precision_score(yt, ypr)
        except: m["pr_auc"] = 0.0
    else:
        m["auc_roc"] = 0.5; m["pr_auc"] = 0.0
    tn,fp,fn,tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    m["TP"]=int(tp); m["TN"]=int(tn); m["FP"]=int(fp); m["FN"]=int(fn)
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    m["sensitivity"]=round(sens,4); m["specificity"]=round(spec,4)
    m["gmean"] = np.sqrt(sens*spec)
    return m


def tune_threshold(yt, yp, n=50):
    if len(yt)==0 or len(np.unique(yt))<2: return THRESHOLD_DEFAULT
    best_mcc, best_t = -2, THRESHOLD_DEFAULT
    for t in np.linspace(0.1, 0.9, n):
        mcc = matthews_corrcoef(yt, (yp>=t).astype(int))
        if mcc > best_mcc: best_mcc, best_t = mcc, t
    return best_t


def run(frag_path, pair_path, output_dir=None):
    t0 = time.time()
    print(f"Loading fragment CSV: {frag_path}")
    frag_df = pd.read_csv(frag_path)
    print(f"  Rows: {len(frag_df):,}, Revisions: {frag_df['revision'].nunique()}, gcids: {frag_df['gcid'].nunique()}")
    print(f"Loading pair CSV: {pair_path}")
    pair_df = pd.read_csv(pair_path)
    print(f"  Rows: {len(pair_df):,}, Columns: {len(pair_df.columns)}")

    # Filter inactive gcids
    ever = frag_df.groupby("gcid")["totalChanges"].max()
    active = set(ever[ever > 0].index)
    frag_df = frag_df[frag_df["gcid"].isin(active)].copy()
    pair_df = pair_df[pair_df["gcid1"].isin(active) | pair_df["gcid2"].isin(active)].copy()

    tag = f"{Path(frag_path).parent.parent.parent.name}_{Path(frag_path).stem.replace('_rev_fragment','')}"
    system_name = Path(frag_path).parent.parent.parent.name  # e.g. "Ctags"
    print(f"  Tag: {tag}")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results" / "predict_standing" / system_name
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    # Index
    frag_index = {r: g for r, g in frag_df.groupby("revision")}
    pair_index = {r: g for r, g in pair_df.groupby("revision")}
    frag_revs = {g: sorted(grp["revision"].tolist()) for g, grp in frag_df.groupby("gcid")}
    pair_revs = sorted(pair_df["revision"].unique().tolist())
    change_revisions = sorted(frag_df.loc[frag_df["changeType"].str.upper()=="M","revision"].unique())
    print(f"  {len(change_revisions)} change revisions, warmup={WARMUP_REVS}, REFIT_EVERY={REFIT_EVERY}\n")

    models = get_models()
    model_names = list(models.keys())
    print(f"Models: {model_names}\n")

    # State per model
    vlog = []
    per_rev = []
    gdata = {mn: {"y_true":[],"y_pred":[],"y_proba":[]} for mn in model_names}
    calib = {mn: {"yt":[],"yp":[]} for mn in model_names}
    thresholds = {mn: THRESHOLD_DEFAULT for mn in model_names}
    fitted = {mn: False for mn in model_names}
    last_refit = {mn: 0 for mn in model_names}
    all_X, all_y = [], []

    for idx, R in enumerate(change_revisions):
        alive = frag_index.get(R)
        if alive is None: continue
        changed = alive[alive["changeType"].str.upper() == "M"]

        rev_vecs, rev_labels, rev_gcids = [], [], []
        for _, row in changed.iterrows():
            g = int(row["gcid"])
            xv = build_fv(g, R, frag_index, pair_index, frag_revs, pair_revs)
            if xv is None: continue
            ay = actual_label(g, R, pair_index)
            rev_vecs.append(xv); rev_labels.append(ay); rev_gcids.append(g)

        if not rev_vecs: continue

        Xr = np.array(rev_vecs); yr = np.array(rev_labels)

        for mn in model_names:
            model = models[mn]
            preds, probas = None, None

            if fitted[mn] and idx >= WARMUP_REVS:
                try:
                    probas = model.predict_proba(Xr)[:, 1]
                    preds = (probas >= thresholds[mn]).astype(int)
                except: pass

            for j in range(len(rev_gcids)):
                entry = {"model": mn, "revision": R, "gcid": rev_gcids[j],
                         "actual_label": rev_labels[j],
                         "pred_label": int(preds[j]) if preds is not None else None,
                         "pred_proba": round(float(probas[j]),4) if probas is not None else None,
                         "correct": int(preds[j]==rev_labels[j]) if preds is not None else None,
                         "threshold": round(thresholds[mn],3)}
                vlog.append(entry)

            if preds is not None:
                rm = compute_metrics(yr, preds, probas)
                rm["model"]=mn; rm["revision"]=R; rm["threshold"]=round(thresholds[mn],3)
                rm["train_size"]=len(all_y)
                per_rev.append(rm)
                gdata[mn]["y_true"].extend(yr.tolist())
                gdata[mn]["y_pred"].extend(preds.tolist())
                gdata[mn]["y_proba"].extend(probas.tolist())
                calib[mn]["yt"].extend(yr.tolist())
                calib[mn]["yp"].extend(probas.tolist())
                if len(calib[mn]["yt"]) > 200:
                    calib[mn]["yt"] = calib[mn]["yt"][-200:]
                    calib[mn]["yp"] = calib[mn]["yp"][-200:]

        # Add to training pool
        all_X.extend(rev_vecs); all_y.extend(rev_labels)
        n_lab = len(all_y)

        # Retrain all models
        if n_lab >= MIN_TRAIN and len(np.unique(all_y)) >= 2:
            Xa, ya = np.array(all_X), np.array(all_y)
            for mn in model_names:
                if n_lab - last_refit[mn] >= REFIT_EVERY or not fitted[mn]:
                    try:
                        m = models[mn]
                        if mn == "XGBoost":
                            neg=(ya==0).sum(); pos=(ya==1).sum()
                            m.set_params(scale_pos_weight=neg/pos if pos>0 else 1)
                        m.fit(Xa, ya)
                        fitted[mn] = True; last_refit[mn] = n_lab
                        cy = np.array(calib[mn]["yt"][-CALIB_WINDOW:])
                        cp = np.array(calib[mn]["yp"][-CALIB_WINDOW:])
                        if len(cy) >= 10 and len(np.unique(cy)) >= 2:
                            thresholds[mn] = tune_threshold(cy, cp)
                    except Exception as e:
                        print(f"  [WARN] {mn} fit failed at rev {R}: {e}")

        if (idx+1) % 25 == 0:
            best_mn, best_mcc = "", -2
            for mn in model_names:
                if gdata[mn]["y_true"]:
                    mcc = matthews_corrcoef(gdata[mn]["y_true"], gdata[mn]["y_pred"])
                    if mcc > best_mcc: best_mcc, best_mn = mcc, mn
            print(f"  Rev {R} ({idx+1}/{len(change_revisions)}) events={n_lab} best={best_mn} MCC={best_mcc:.3f}")

    # ── GLOBAL RESULTS ──
    W = 70
    print(f"\n{'='*W}\n  GLOBAL RESULTS: {tag}\n{'='*W}")
    global_rows = []
    for mn in model_names:
        d = gdata[mn]
        if not d["y_true"]: continue
        yt = np.array(d["y_true"]); yp = np.array(d["y_pred"]); ypr = np.array(d["y_proba"])
        gm = compute_metrics(yt, yp, ypr)
        gm["model"] = mn; gm["total_predictions"] = len(yt)
        global_rows.append(gm)
        print(f"\n  {mn}:")
        print(f"    MCC={gm['mcc']:.4f}  AUC-ROC={gm['auc_roc']:.4f}  PR-AUC={gm['pr_auc']:.4f}")
        print(f"    Bal.Acc={gm['balanced_acc']:.4f}  G-mean={gm['gmean']:.4f}  F1={gm['f1']:.4f}")
        print(f"    Sens={gm['sensitivity']}  Spec={gm['specificity']}")
        print(f"    TP={gm['TP']} TN={gm['TN']} FP={gm['FP']} FN={gm['FN']}  n={gm['total_predictions']}")

    # ── SAVE ──
    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(output_dir / f"standing_{tag}_global_metrics.csv", index=False)
    pd.DataFrame(per_rev).to_csv(output_dir / f"standing_{tag}_per_revision.csv", index=False)
    pd.DataFrame(vlog).to_csv(output_dir / f"standing_{tag}_verification_log.csv", index=False)
    print(f"\n  Results saved to {output_dir}")

    # ── PLOTS ──
    try:
        from plots import (plot_rolling_mcc, plot_confusion_matrices, plot_roc_curves,
                           plot_pr_curves, plot_feature_importance, plot_model_comparison,
                           plot_label_distribution)
        rev_df = pd.DataFrame(per_rev)
        plot_rolling_mcc(rev_df, model_names, tag, output_dir)
        plot_confusion_matrices(gdata, model_names, tag, output_dir)
        plot_roc_curves(gdata, model_names, tag, output_dir)
        plot_pr_curves(gdata, model_names, tag, output_dir)
        plot_feature_importance(models, FEATURE_COLS, tag, output_dir)
        plot_model_comparison(global_df, tag, output_dir)
        all_yt = [e["actual_label"] for e in vlog if e["model"]==model_names[0]]
        plot_label_distribution(all_yt, tag, output_dir)
    except Exception as e:
        print(f"  [WARN] Plotting error: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*W}\n  DONE ({elapsed:.1f}s)\n{'='*W}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Standing prediction for clone fragments")
    p.add_argument("--frag", required=True, help="Path to rev_fragment.csv")
    p.add_argument("--pair", required=True, help="Path to rev_pair.csv")
    p.add_argument("--output", default=None, help="Output directory")
    a = p.parse_args()
    run(a.frag, a.pair, a.output)
