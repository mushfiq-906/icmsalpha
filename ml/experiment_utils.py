"""
experiment_utils.py -- Shared engine for all reviewer-response experiments.
Provides: data loading, feature groups, walk-forward loop, metrics.
"""
import sys, bisect, warnings, os, time
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
from pathlib import Path
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
    from catboost import CatBoostClassifier; HAS_CB = True
except ImportError: HAS_CB = False

# ── FEATURE GROUPS ──────────────────────────────────────────────────────────
DECAY_HORIZONS = [10, 20, 30, 50, 75]

FRAG_NUMERIC = [
    "nlines","totalChanges","totalUnchanged","stabilityIndex",
    "changeProneness","lifespan","activePairs","meanWCS","maxWCS",
    "minWCS","stronglyCoupledPairs","decoupledPairs",
]
FRAG_SIDE_COLS = ["churn","distinctAuthors","majorAuthorProp","minorAuthorCount"]
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
    [f"mean_ir_decay_h{h}"   for h in DECAY_HORIZONS] +
    [f"mean_cs_decay_h{h}"   for h in DECAY_HORIZONS] +
    [f"mean_spcp_decay_h{h}" for h in DECAY_HORIZONS]
)
ALL_FEATURE_COLS = FRAG_NUMERIC + FRAG_SIDE_COLS + PAIR_AGG_COLS + PAIR_ENRICHED + DECAY_AGG

# Feature group ablation variants
ABLATION_VARIANTS = {
    "full":        ALL_FEATURE_COLS,
    "no_fragment": FRAG_SIDE_COLS + PAIR_AGG_COLS + PAIR_ENRICHED + DECAY_AGG,
    "no_process":  FRAG_NUMERIC   + PAIR_AGG_COLS + PAIR_ENRICHED + DECAY_AGG,
    "no_pair":     FRAG_NUMERIC   + FRAG_SIDE_COLS                + DECAY_AGG,
    "no_decay":    FRAG_NUMERIC   + FRAG_SIDE_COLS + PAIR_AGG_COLS + PAIR_ENRICHED,
}

SENTINEL_AGE = 9999

# Pair CSV dtype map — int32/float32 cuts memory ~50% for large systems (e.g. tuxguitar 4.6M rows)
_PAIR_INT32 = ["revision","gcid1","gcid2","classId","coChangeCount",
               "gcid1Independent","gcid2Independent","lastCoChangeAge",
               "lastSoloChangeAge","similarity","depth","sameFile",
               "sameAuthor","classSize","isSpcp","wcsRecent",
               "soloEventsAfterLastCoChange","nlines1","nlines2",
               "noc1","noc2","fileAge1","fileAge2","churn1","churn2",
               "distinctAuthors1","distinctAuthors2",
               "minorAuthorCount1","minorAuthorCount2"]
_PAIR_FLOAT32 = ["weightedCouplingStrength","couplingTrend",
                 "majorAuthorProp1","majorAuthorProp2"] + [
    f"{p}_h{h}" for p in ["ir_decay","cs_decay","spcp_decay"]
    for h in [10,20,30,50,75]
]


def _make_pair_dtype(path):
    """Build dtype dict: safe int64->int32, float64->float32, skip strings.
    Reads 100 rows first to detect actual types to avoid casting errors."""
    import csv
    # Get column names
    with open(path, "r") as f:
        cols = next(csv.reader(f))
    # Sample 500 rows to detect real dtypes
    sample = pd.read_csv(path, nrows=500)
    dtype = {}
    for c in cols:
        if c not in sample.columns: continue
        dt = sample[c].dtype
        if dt == "float64":
            dtype[c] = "float32"
        elif dt == "int64":
            # Only cast to int32 if values are safely within range
            try:
                mn, mx = int(sample[c].min()), int(sample[c].max())
                if -2147483648 <= mn and mx <= 2147483647:
                    dtype[c] = "int32"
            except: pass
    return dtype


# ── DATA LOADING ─────────────────────────────────────────────────────────────
def load_data(frag_path, pair_path):
    print(f"Loading fragment: {frag_path}")
    frag_df = pd.read_csv(frag_path)
    print(f"  Rows: {len(frag_df):,}  revisions: {frag_df['revision'].nunique()}  gcids: {frag_df['gcid'].nunique()}")

    ever   = frag_df.groupby("gcid")["totalChanges"].max()
    active = set(ever[ever > 0].index)
    frag_df = frag_df[frag_df["gcid"].isin(active)].reset_index(drop=True)

    print(f"Loading pair:     {pair_path}")
    pair_dtype = _make_pair_dtype(pair_path)
    pair_df = pd.read_csv(pair_path, dtype=pair_dtype)
    print(f"  Rows: {len(pair_df):,}  columns: {len(pair_df.columns)}")

    # Filter to active gcids — use reset_index instead of .copy() to avoid
    # pandas block-consolidation OOM on large DataFrames (e.g. tuxguitar 4.6M rows)
    mask    = pair_df["gcid1"].isin(active) | pair_df["gcid2"].isin(active)
    pair_df = pair_df[mask].reset_index(drop=True)
    print(f"  After filter: {len(pair_df):,} rows")

    frag_index  = {r: g for r, g in frag_df.groupby("revision")}
    pair_index  = {r: g for r, g in pair_df.groupby("revision")}
    frag_revs   = {g: sorted(grp["revision"].tolist()) for g, grp in frag_df.groupby("gcid")}
    pair_revs   = sorted(pair_df["revision"].unique().tolist())
    change_revs = sorted(frag_df.loc[frag_df["changeType"].str.upper()=="M","revision"].unique())
    print(f"  {len(change_revs)} change-revisions\n")
    return frag_df, pair_df, frag_index, pair_index, frag_revs, pair_revs, change_revs


# ── FEATURE BUILDING ──────────────────────────────────────────────────────────
def _safe(df, col, default=0.0):
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


def _empty_pair_feats():
    f = {c: 0.0 for c in PAIR_AGG_COLS + PAIR_ENRICHED + DECAY_AGG}
    f["min_last_co_change_age"]   = SENTINEL_AGE
    f["min_last_solo_change_age"] = SENTINEL_AGE
    f["class_size"] = 1; f["mean_class_size"] = 1.0
    return f


def aggregate_pairs(pair_at, gcid):
    mask  = (pair_at["gcid1"] == gcid) | (pair_at["gcid2"] == gcid)
    pairs = pair_at[mask]
    if len(pairs) == 0: return None
    own_indep = np.where(pairs["gcid1"].values == gcid,
                         pairs["gcid1Independent"].values,
                         pairs["gcid2Independent"].values)
    last_co   = pairs["lastCoChangeAge"].replace(-1, SENTINEL_AGE).values
    last_solo = pairs["lastSoloChangeAge"].replace(-1, SENTINEL_AGE).values
    o = {
        "pair_count_alive":         len(pairs),
        "max_co_change_count":      int(pairs["coChangeCount"].max()),
        "sum_co_change_count":      int(pairs["coChangeCount"].sum()),
        "max_own_independent":      int(own_indep.max()),
        "sum_own_independent":      int(own_indep.sum()),
        "mean_coupling_trend":      float(pairs["couplingTrend"].mean()),
        "min_coupling_trend":       float(pairs["couplingTrend"].min()),
        "min_last_co_change_age":   int(last_co.min()),
        "min_last_solo_change_age": int(last_solo.min()),
        "mean_similarity":          float(pairs["similarity"].mean()),
        "max_similarity":           float(pairs["similarity"].max()),
        "frac_same_file":           float(pairs["sameFile"].mean()),
        "mean_depth":               float(pairs["depth"].mean()),
    }
    o["mean_wcs_recent"]  = float(_safe(pairs,"wcsRecent",0).mean())
    o["max_wcs_recent"]   = float(_safe(pairs,"wcsRecent",0).max())
    o["min_wcs_recent"]   = float(_safe(pairs,"wcsRecent",0).min())
    o["max_solo_after_last_cochange"] = int(_safe(pairs,"soloEventsAfterLastCoChange",0).max())
    o["frac_is_spcp"]     = float(_safe(pairs,"isSpcp",0).mean())
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


def build_fv(gcid, R, frag_index, pair_index, frag_revs, pair_revs):
    """Returns (full_52d_vec, pair_df_at_prev_rev) or (None, None)."""
    grevs = frag_revs.get(gcid)
    if not grevs: return None, None
    idx  = bisect.bisect_left(grevs, R)
    frev = grevs[idx-1] if idx > 0 else None
    if frev is None: return None, None
    rows = frag_index[frev]
    rows = rows[rows["gcid"] == gcid]
    if len(rows) == 0: return None, None
    row  = rows.iloc[0]
    pidx = bisect.bisect_left(pair_revs, R)
    prev = pair_revs[pidx-1] if pidx > 0 else None
    pat  = pair_index.get(prev) if prev else None
    pf   = aggregate_pairs(pat, gcid) if pat is not None else None
    if pf is None: pf = _empty_pair_feats()
    sf   = extract_side_feats(pat, gcid)
    full_vec = (
        [float(row[c]) for c in FRAG_NUMERIC] +
        [sf[c]         for c in FRAG_SIDE_COLS] +
        [pf[c]         for c in PAIR_AGG_COLS] +
        [pf[c]         for c in PAIR_ENRICHED] +
        [pf[c]         for c in DECAY_AGG]
    )
    return np.array(full_vec, dtype=float), pat


# ── LABELING FUNCTIONS ────────────────────────────────────────────────────────
def actual_label(gcid, rev, pair_index, frag_index=None):
    pat = pair_index.get(rev)
    if pat is None: return 0
    m1 = (pat["gcid1"]==gcid) & (pat["changeType2"].str.upper()=="M")
    m2 = (pat["gcid2"]==gcid) & (pat["changeType1"].str.upper()=="M")
    return 1 if (m1.any() or m2.any()) else 0


def actual_label_strict_module(gcid, rev, pair_index, frag_index):
    """Stricter: sibling must be in the same package/directory."""
    pat = pair_index.get(rev)
    if pat is None: return 0
    fr  = frag_index.get(rev)
    if fr is None: return actual_label(gcid, rev, pair_index)
    gcid_rows = fr[fr["gcid"]==gcid]
    if len(gcid_rows) == 0: return actual_label(gcid, rev, pair_index)
    gcid_file = str(gcid_rows.iloc[0]["filePath"]).replace("\\","/")
    gcid_pkg  = "/".join(gcid_file.split("/")[:-1])
    m1 = pat[(pat["gcid1"]==gcid) & (pat["changeType2"].str.upper()=="M")]
    m2 = pat[(pat["gcid2"]==gcid) & (pat["changeType1"].str.upper()=="M")]
    sibling_ids = list(m1["gcid2"].astype(int)) + list(m2["gcid1"].astype(int))
    for sg in sibling_ids:
        sg_rows = fr[fr["gcid"]==sg]
        if len(sg_rows) > 0:
            sg_file = str(sg_rows.iloc[0]["filePath"]).replace("\\","/")
            sg_pkg  = "/".join(sg_file.split("/")[:-1])
            if sg_pkg == gcid_pkg:
                return 1
    return 0


def make_excl_auto_label_fn(excluded_revisions, base_label_fn=None):
    """Returns a label function that returns None for excluded revisions."""
    excl = set(excluded_revisions)
    base = base_label_fn or actual_label
    def fn(gcid, rev, pair_index, frag_index):
        if rev in excl: return None
        return base(gcid, rev, pair_index, frag_index)
    return fn


# ── BASELINES ─────────────────────────────────────────────────────────────────
def wcs_threshold_predict(gcid, pat, threshold=0.5):
    """Predict dependent if maxWCS >= threshold."""
    if pat is None: return 0
    mask  = (pat["gcid1"]==gcid) | (pat["gcid2"]==gcid)
    pairs = pat[mask]
    if len(pairs) == 0: return 0
    col = "weightedCouplingStrength"
    max_wcs = float(pairs[col].max()) if col in pairs.columns else 0.0
    return 1 if max_wcs >= threshold else 0


def cochange_rate_predict(gcid, pat, threshold=0.5):
    """Predict dependent if historical co-change rate >= threshold."""
    if pat is None: return 0
    mask  = (pat["gcid1"]==gcid) | (pat["gcid2"]==gcid)
    pairs = pat[mask]
    if len(pairs) == 0: return 0
    own_indep = np.where(pairs["gcid1"].values==gcid,
                         pairs["gcid1Independent"].values,
                         pairs["gcid2Independent"].values)
    total = pairs["coChangeCount"].values + own_indep
    rates = np.where(total > 0, pairs["coChangeCount"].values / total, 0.0)
    return 1 if float(rates.mean()) >= threshold else 0


# ── METRICS ────────────────────────────────────────────────────────────────────
def compute_metrics(yt, yp, ypr=None):
    yt, yp = np.array(yt), np.array(yp)
    m = {"n": len(yt), "n_pos": int(yt.sum()), "n_neg": int((yt==0).sum())}
    if len(np.unique(yt)) < 2:
        m.update(mcc=0.0, auc_roc=0.5, pr_auc=0.0, balanced_acc=0.5, f1=0.0,
                 sensitivity=0.0, specificity=0.0, TP=0, TN=0, FP=0, FN=0, gmean=0.0)
        return m
    m["mcc"]          = matthews_corrcoef(yt, yp)
    m["balanced_acc"] = balanced_accuracy_score(yt, yp)
    m["f1"]           = f1_score(yt, yp, zero_division=0)
    if ypr is not None and len(np.unique(ypr)) > 1:
        try:    m["auc_roc"] = roc_auc_score(yt, ypr)
        except: m["auc_roc"] = 0.5
        try:    m["pr_auc"]  = average_precision_score(yt, ypr)
        except: m["pr_auc"]  = 0.0
    else:
        m["auc_roc"] = 0.5; m["pr_auc"] = 0.0
    tn,fp,fn,tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    m["TP"]=int(tp); m["TN"]=int(tn); m["FP"]=int(fp); m["FN"]=int(fn)
    sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    m["sensitivity"] = round(sens,4); m["specificity"] = round(spec,4)
    m["gmean"] = float(np.sqrt(sens*spec))
    return m


def tune_threshold(yt, yp, n=50):
    if len(yt)==0 or len(np.unique(yt))<2: return 0.5
    best_mcc, best_t = -2.0, 0.5
    for t in np.linspace(0.1, 0.9, n):
        mcc = matthews_corrcoef(yt, (yp>=t).astype(int))
        if mcc > best_mcc: best_mcc, best_t = mcc, t
    return best_t


# ── MODEL FACTORY ─────────────────────────────────────────────────────────────
def get_models(seed=42, best_only=False, n_jobs=-1):
    """Return classifiers. If best_only=True, return only RandomForest."""
    m = {}
    m["RandomForest"] = RandomForestClassifier(
        n_estimators=700, max_depth=15, min_samples_leaf=10,
        class_weight="balanced", random_state=seed, n_jobs=n_jobs)
    if not best_only:
        if HAS_LGBM:
            m["LightGBM"] = LGBMClassifier(
                n_estimators=700, max_depth=8, learning_rate=0.04,
                num_leaves=31, min_child_samples=10, is_unbalance=True,
                random_state=seed, verbose=-1, n_jobs=-1)
        if HAS_XGB:
            m["XGBoost"] = XGBClassifier(
                n_estimators=700, max_depth=8, learning_rate=0.04,
                min_child_weight=5, eval_metric="logloss",
                random_state=seed, verbosity=0, n_jobs=-1)
        if HAS_CB:
            import tempfile
            m["CatBoost"] = CatBoostClassifier(
                iterations=700, depth=8, learning_rate=0.04,
                auto_class_weights="Balanced", random_seed=seed,
                verbose=0, thread_count=-1,
                train_dir=os.path.join(tempfile.gettempdir(), f"catboost_exp_{seed}"))
    return m


def precompute_stream(frag_index, pair_index, frag_revs, pair_revs, change_revs, label_fn=None):
    """Precompute feature vectors and labels once per revision to avoid redundant DataFrame indexing in multiple passes."""
    if label_fn is None: label_fn = actual_label
    stream = []
    for R in change_revs:
        alive = frag_index.get(R)
        if alive is None: continue
        changed = alive[alive["changeType"].str.upper() == "M"]
        rev_vecs, rev_labels, rev_gcids, rev_pats = [], [], [], []
        for _, row in changed.iterrows():
            g  = int(row["gcid"])
            xv, pat = build_fv(g, R, frag_index, pair_index, frag_revs, pair_revs)
            if xv is None: continue
            ay = label_fn(g, R, pair_index, frag_index)
            if ay is None: continue
            rev_vecs.append(xv); rev_labels.append(ay)
            rev_gcids.append(g);  rev_pats.append(pat)
        if rev_vecs:
            Xfull = np.array(rev_vecs, dtype=np.float32)
            yr    = np.array(rev_labels, dtype=np.int32)
            stream.append((R, Xfull, yr, rev_gcids, rev_pats))
    return stream


# ── CORE WALK-FORWARD ENGINE ──────────────────────────────────────────────────
def walk_forward(
    frag_index, pair_index, frag_revs, pair_revs, change_revs,
    feature_cols=None,
    label_fn=None,
    models=None,
    baselines=None,
    warmup_revs=5,
    min_train=30,
    calib_window=50,
    seed=42,
    verbose=True,
    stream=None,
):
    """
    Core walk-forward loop. Returns dict of {model_name: metrics_dict}.
    baselines: dict {name: callable(gcid, pat, threshold) -> int}
    label_fn:  callable(gcid, rev, pair_index, frag_index) -> int or None
    stream:    optional precomputed list of (R, Xfull, yr, rev_gcids, rev_pats)
    """
    if feature_cols is None: feature_cols = ALL_FEATURE_COLS
    if label_fn    is None: label_fn = actual_label
    if models      is None: models   = get_models(seed)
    if baselines   is None: baselines = {}

    # Map feature_cols to indices in ALL_FEATURE_COLS
    feat_idx = [ALL_FEATURE_COLS.index(c) for c in feature_cols]

    model_names    = list(models.keys())
    baseline_names = list(baselines.keys())
    all_names      = model_names + baseline_names

    gdata         = {n: {"y_true":[],"y_pred":[],"y_proba":[]} for n in all_names}
    calib         = {n: {"yt":[],"yp":[]}                       for n in model_names}
    thresholds    = {n: 0.5 for n in model_names}
    bl_thresholds = {n: 0.5 for n in baseline_names}
    bl_calib      = {n: {"yt":[],"yp":[]} for n in baseline_names}
    fitted        = {n: False for n in model_names}
    last_refit    = {n: 0     for n in model_names}
    all_X, all_y  = [], []

    t0 = time.time()

    def _get_rev_items():
        if stream is not None:
            for item in stream:
                if len(item) == 5:
                    R, Xfull, yr, r_gcids, r_pats = item
                else:
                    R, Xfull, yr = item[:3]
                    r_gcids, r_pats = [], []
                yield R, Xfull, yr, r_gcids, r_pats, [Xfull[j] for j in range(len(Xfull))], yr.tolist()
        else:
            for R in change_revs:
                alive = frag_index.get(R)
                if alive is None: continue
                changed = alive[alive["changeType"].str.upper() == "M"]
                rev_vecs, rev_labels, rev_gcids, rev_pats = [], [], [], []
                for _, row in changed.iterrows():
                    g  = int(row["gcid"])
                    xv, pat = build_fv(g, R, frag_index, pair_index, frag_revs, pair_revs)
                    if xv is None: continue
                    ay = label_fn(g, R, pair_index, frag_index)
                    if ay is None: continue  # excluded event
                    rev_vecs.append(xv); rev_labels.append(ay)
                    rev_gcids.append(g);  rev_pats.append(pat)
                if rev_vecs:
                    Xfull = np.array(rev_vecs)
                    yr    = np.array(rev_labels)
                    yield R, Xfull, yr, rev_gcids, rev_pats, rev_vecs, rev_labels

    num_revs = len(stream) if stream is not None else len(change_revs)
    for idx, (R, Xfull, yr, rev_gcids, rev_pats, rev_vecs, rev_labels) in enumerate(_get_rev_items()):
        Xr    = Xfull[:, feat_idx]

        # ML models
        for mn in model_names:
            if fitted[mn] and idx >= warmup_revs:
                try:
                    Xr_df  = pd.DataFrame(Xr, columns=feature_cols)
                    probas = models[mn].predict_proba(Xr_df)[:,1]
                    preds  = (probas >= thresholds[mn]).astype(int)
                    gdata[mn]["y_true"].extend(yr.tolist())
                    gdata[mn]["y_pred"].extend(preds.tolist())
                    gdata[mn]["y_proba"].extend(probas.tolist())
                    calib[mn]["yt"].extend(yr.tolist())
                    calib[mn]["yp"].extend(probas.tolist())
                    if len(calib[mn]["yt"]) > 200:
                        calib[mn]["yt"] = calib[mn]["yt"][-200:]
                        calib[mn]["yp"] = calib[mn]["yp"][-200:]
                except: pass

        # Baselines (threshold-free or adaptive)
        if idx >= warmup_revs:
            for bn, bl_fn in baselines.items():
                bt = bl_thresholds[bn]
                preds_bl = [bl_fn(rev_gcids[j], rev_pats[j], bt) for j in range(len(rev_gcids))]
                # Use WCS value as "proba" for AUC
                probas_bl = []
                for j in range(len(rev_gcids)):
                    pat_j = rev_pats[j]
                    if pat_j is not None:
                        mask = (pat_j["gcid1"]==rev_gcids[j])|(pat_j["gcid2"]==rev_gcids[j])
                        pairs_j = pat_j[mask]
                        if len(pairs_j)>0 and "weightedCouplingStrength" in pairs_j.columns:
                            probas_bl.append(float(pairs_j["weightedCouplingStrength"].max()))
                        else: probas_bl.append(float(preds_bl[j]))
                    else: probas_bl.append(float(preds_bl[j]))
                gdata[bn]["y_true"].extend(yr.tolist())
                gdata[bn]["y_pred"].extend(preds_bl)
                gdata[bn]["y_proba"].extend(probas_bl)
                bl_calib[bn]["yt"].extend(yr.tolist())
                bl_calib[bn]["yp"].extend(probas_bl)
                if len(bl_calib[bn]["yt"]) > calib_window:
                    bl_calib[bn]["yt"] = bl_calib[bn]["yt"][-calib_window:]
                    bl_calib[bn]["yp"] = bl_calib[bn]["yp"][-calib_window:]
                # Adapt threshold for WCS-threshold baseline
                cy = np.array(bl_calib[bn]["yt"])
                cp = np.array(bl_calib[bn]["yp"])
                if len(cy) >= 10 and len(np.unique(cy)) >= 2:
                    bl_thresholds[bn] = tune_threshold(cy, cp)

        # Add to pool and refit
        all_X.extend(rev_vecs); all_y.extend(rev_labels)
        n_lab = len(all_y)
        if n_lab >= min_train and len(np.unique(all_y)) >= 2:
            Xa    = np.array(all_X)[:, feat_idx]
            ya    = np.array(all_y)
            Xa_df = pd.DataFrame(Xa, columns=feature_cols)
            for mn in model_names:
                if n_lab - last_refit[mn] >= 1 or not fitted[mn]:
                    try:
                        m = models[mn]
                        if mn == "XGBoost":
                            neg=(ya==0).sum(); pos=(ya==1).sum()
                            m.set_params(scale_pos_weight=neg/pos if pos>0 else 1)
                        m.fit(Xa_df, ya)
                        fitted[mn] = True; last_refit[mn] = n_lab
                        cy = np.array(calib[mn]["yt"][-calib_window:])
                        cp = np.array(calib[mn]["yp"][-calib_window:])
                        if len(cy) >= 10 and len(np.unique(cy)) >= 2:
                            thresholds[mn] = tune_threshold(cy, cp)
                    except Exception as e:
                        if verbose: print(f"  [WARN] {mn} fit failed rev {R}: {e}")

        if verbose and (idx+1) % 50 == 0:
            best_mn, best_mcc = "", -2.0
            for mn in model_names:
                if gdata[mn]["y_true"]:
                    mcc = matthews_corrcoef(gdata[mn]["y_true"], gdata[mn]["y_pred"])
                    if mcc > best_mcc: best_mcc, best_mn = mcc, mn
            print(f"  [{int(time.time()-t0)}s] Rev {R} ({idx+1}/{num_revs}) "
                  f"pool={n_lab}  best={best_mn} MCC={best_mcc:.3f}", flush=True)

    # Aggregate final metrics
    results = {}
    for n in all_names:
        d = gdata[n]
        if not d["y_true"]: continue
        gm = compute_metrics(d["y_true"], d["y_pred"], d["y_proba"])
        gm["model"] = n; gm["total_predictions"] = len(d["y_true"])
        results[n] = gm
    return results
