"""
run_parallel_experiments.py
===========================
High-Performance Master Runner implementing:
  - Strategy 2: Precomputed Feature Stream Caching (zero repeated lookups/bisects)
  - Strategy 3: Multi-Core ProcessPool Parallelism across Seeds, Parameter Sweeps, and Ablations
Specifically tailored for multi-core AMD Ryzen 9 7950X architecture.

Usage:
  python run_parallel_experiments.py --systems dnsjava Jmol --workers 4
"""
import sys, os, re, time, warnings, argparse, pickle
import concurrent.futures
from pathlib import Path

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from experiment_utils import (
    load_data, walk_forward, get_models, compute_metrics,
    actual_label, actual_label_strict_module, make_excl_auto_label_fn,
    wcs_threshold_predict, cochange_rate_predict, precompute_stream,
    ALL_FEATURE_COLS, ABLATION_VARIANTS,
)

# -- SYSTEM PATHS --------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent / "WorkFolder"
NICAD_BASE = Path(r"C:\Users\CSE-AI-Lab\Desktop\Thesis\Nicad\systems")

SYSTEMS = {
    "tuxguitar": {
        "frag": BASE / "tuxguitar" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_fragment.csv",
        "pair": BASE / "tuxguitar" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_pair.csv",
        "commit_log": None,
    },
    "Ctags": {
        "frag": BASE / "Ctags" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_fragment.csv",
        "pair": BASE / "Ctags" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_pair.csv",
        "commit_log": None,
    },
    "dnsjava": {
        "frag": BASE / "dnsjava" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_fragment.csv",
        "pair": BASE / "dnsjava" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_pair.csv",
        "commit_log": NICAD_BASE / "dnsjava" / "commit_logs.txt",
    },
    "Jmol": {
        "frag": BASE / "Jmol" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_fragment.csv",
        "pair": BASE / "Jmol" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_pair.csv",
        "commit_log": NICAD_BASE / "Jmol" / "commit_logs.txt",
    },
}

SEEDS = [42, 0, 1, 7, 13, 17, 21, 37, 99, 123]

PARAM_GRID = [
    # (warmup_revs, min_train, calib_window, label)
    (5,  30,  50,  "default"),
    (3,  30,  50,  "warmup=3"),
    (10, 30,  50,  "warmup=10"),
    (5,  10,  50,  "min_train=10"),
    (5,  50,  50,  "min_train=50"),
    (5,  30,  20,  "calib=20"),
    (5,  30, 100,  "calib=100"),
    (5,  30, 9999, "calib=inf"),
]

AUTO_PATTERNS = re.compile(
    r'\breformat\b|\bformatting\b|\bindent(ation)?\b|\bwhitespace\b|'
    r'\bmerge\b|\bbump.version\b|\brelease\b|\bupdate.version\b|'
    r'^\*\*\*.empty.log|\binitial.revision\b|\bimport.of\b|'
    r'\bauto.?generated\b|\bno.message\b',
    re.IGNORECASE
)

def load_automated_revisions(commit_log_path):
    if commit_log_path is None or not Path(commit_log_path).exists():
        return set()
    rows = []
    with open(commit_log_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 0: continue
            parts = line.rstrip().split(" | ")
            if len(parts) < 7: continue
            try:
                rev = int(parts[0].strip())
                msg = " | ".join(parts[4:]).strip()
                rows.append((rev, msg))
            except: continue
    auto_revs = set()
    for rev, msg in rows:
        if AUTO_PATTERNS.search(msg):
            auto_revs.add(rev)
    print(f"  Commit log: {len(rows)} entries, {len(auto_revs)} automated revisions identified", flush=True)
    return auto_revs


def save_results(rows, out_dir, filename):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out_dir / filename
    df.to_csv(path, index=False)
    print(f"  Saved: {path}", flush=True)
    return df


def print_results(results, label=""):
    W = 72
    if label: print(f"\n{'-'*W}\n  {label}\n{'-'*W}", flush=True)
    for name, m in results.items():
        print(f"  {name:20s}  MCC={m.get('mcc',0):.4f}  "
              f"BalAcc={m.get('balanced_acc',0):.4f}  "
              f"AUC={m.get('auc_roc',0):.4f}  "
              f"F1={m.get('f1',0):.4f}  n={m.get('n',0)}", flush=True)


# ==============================================================================
# STRATEGY 2: FEATURE CACHING
# ==============================================================================
def get_or_create_stream(system, data, out_dir):
    cache_path = out_dir / "stream_cache.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                stream_lite = pickle.load(f)
            total_samples = sum(len(yr) for _, _, yr in stream_lite)
            print(f"  [Strategy 2] Loaded feature stream from cache: {len(stream_lite)} revisions, {total_samples} samples.", flush=True)
            return stream_lite
        except Exception as e:
            print(f"  [Strategy 2] Cache load warning: {e}", flush=True)

    frag_idx, pair_idx, frag_revs, pair_revs, change_revs = data
    print(f"  [Strategy 2] Precomputing feature stream for {system} ({len(change_revs)} change-revs)...", flush=True)
    t0 = time.time()
    stream = precompute_stream(frag_idx, pair_idx, frag_revs, pair_revs, change_revs, label_fn=actual_label)
    stream_lite = [(R, Xfull, yr) for (R, Xfull, yr, *_) in stream]
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(stream_lite, f, protocol=pickle.HIGHEST_PROTOCOL)
        total_samples = sum(len(yr) for _, _, yr in stream_lite)
        print(f"  [Strategy 2] Cached {len(stream_lite)} revisions ({total_samples} samples) in {time.time()-t0:.1f}s to {cache_path}", flush=True)
    except Exception as e:
        print(f"  [Strategy 2] Cache save warning: {e}", flush=True)
    return stream_lite


# ==============================================================================
# TOP-LEVEL MULTIPROCESSING WORKERS (FOR WINDOWS PICKLE)
# ==============================================================================
def _worker_eval_seed(args):
    seed, stream_lite, feature_cols, n_jobs = args
    import os, sys, warnings
    warnings.filterwarnings("ignore")
    from experiment_utils import get_models, walk_forward
    models = get_models(seed=seed, best_only=True, n_jobs=n_jobs)
    results = walk_forward(
        None, None, None, None, None,
        feature_cols=feature_cols,
        models=models, baselines={},
        warmup_revs=5, min_train=30, calib_window=50,
        seed=seed, verbose=False, stream=stream_lite,
    )
    return seed, results["RandomForest"]


def _worker_eval_param(args):
    config_tuple, stream_lite, feature_cols, seed, n_jobs = args
    warmup, min_train, calib, label = config_tuple
    import os, sys, warnings
    warnings.filterwarnings("ignore")
    from experiment_utils import get_models, walk_forward
    models = get_models(seed=seed, best_only=True, n_jobs=n_jobs)
    results = walk_forward(
        None, None, None, None, None,
        feature_cols=feature_cols,
        models=models, baselines={},
        warmup_revs=warmup, min_train=min_train, calib_window=min(calib, len(stream_lite)),
        seed=seed, verbose=False, stream=stream_lite,
    )
    return config_tuple, results["RandomForest"]


def _worker_eval_ablation(args):
    variant, feature_cols, stream_lite, seed, n_jobs = args
    import os, sys, warnings
    warnings.filterwarnings("ignore")
    from experiment_utils import get_models, walk_forward
    models = get_models(seed=seed, best_only=True, n_jobs=n_jobs)
    results = walk_forward(
        None, None, None, None, None,
        feature_cols=feature_cols,
        models=models, baselines={},
        warmup_revs=5, min_train=30, calib_window=50,
        seed=seed, verbose=False, stream=stream_lite,
    )
    return variant, len(feature_cols), results["RandomForest"]


# ==============================================================================
# EXPERIMENT 1: BASELINES (R1-C2 / R3-C1)
# ==============================================================================
def run_baselines(system, data, out_dir):
    print(f"\n{'='*72}", flush=True)
    print(f"  EXPERIMENT 1: BASELINES  [{system}]", flush=True)
    print(f"{'='*72}", flush=True)

    base_file = out_dir / "baselines_results.csv"
    if base_file.exists():
        try:
            bdf = pd.read_csv(base_file)
            if "RandomForest" in bdf["model"].values:
                print(f"  -- Baselines already completed in {base_file}, skipping --", flush=True)
                return {r["model"]: r for _, r in bdf.iterrows()}
        except Exception:
            pass

    frag_idx, pair_idx, frag_revs, pair_revs, change_revs = data
    baselines = {
        "WCS-Threshold": wcs_threshold_predict,
        "CoChange-Rate": cochange_rate_predict,
    }
    models = get_models(seed=42, best_only=True, n_jobs=-1)
    results = walk_forward(
        frag_idx, pair_idx, frag_revs, pair_revs, change_revs,
        feature_cols=ALL_FEATURE_COLS,
        label_fn=actual_label,
        models=models,
        baselines=baselines,
        warmup_revs=5, min_train=30, calib_window=50,
        seed=42, verbose=True,
    )
    rows = [{"system": system, "experiment": "baselines", **v} for v in results.values()]
    save_results(rows, out_dir, "baselines_results.csv")
    print_results(results, f"Baselines -- {system}")
    return results


# ==============================================================================
# EXPERIMENT 2: ABLATION (R1-C3) -- STRATEGY 2 + 3 PARALLEL
# ==============================================================================
def run_ablation(system, stream_lite, out_dir, max_workers=3, worker_threads=2):
    print(f"\n{'='*72}", flush=True)
    print(f"  EXPERIMENT 2: FEATURE ABLATION (Strategy 2+3 Parallel) [{system}]", flush=True)
    print(f"{'='*72}", flush=True)

    ablation_file = out_dir / "ablation_results.csv"
    all_rows = []
    if ablation_file.exists():
        try:
            all_rows = pd.read_csv(ablation_file).to_dict("records")
            print(f"  Loaded {len(all_rows)} existing ablation rows from {ablation_file}", flush=True)
        except Exception:
            all_rows = []

    done_variants = {r["variant"] for r in all_rows if "variant" in r}

    # Ensure 'full' is present
    if "full" not in done_variants:
        base_file = out_dir / "baselines_results.csv"
        if base_file.exists():
            bdf = pd.read_csv(base_file)
            rf_full = bdf[bdf["model"]=="RandomForest"]
            if len(rf_full) > 0:
                frow = rf_full.iloc[0].to_dict()
                frow["experiment"] = "ablation"
                frow["variant"] = "full"
                frow["n_features"] = len(ALL_FEATURE_COLS)
                all_rows.append(frow)
                done_variants.add("full")
                save_results(all_rows, out_dir, "ablation_results.csv")
                print(f"  -- Variant: full ({len(ALL_FEATURE_COLS)} features) [Loaded from baselines: MCC={float(frow.get('mcc',0)):.4f}] --", flush=True)

    pending_variants = [(v, cols) for v, cols in ABLATION_VARIANTS.items() if v not in done_variants]
    if not pending_variants:
        print(f"  -- All ablation variants already completed --", flush=True)
    else:
        print(f"  Evaluating {len(pending_variants)} variants with {max_workers} parallel workers...", flush=True)
        tasks = [(v, cols, stream_lite, 42, worker_threads) for v, cols in pending_variants]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker_eval_ablation, t): t[0] for t in tasks}
            for fut in concurrent.futures.as_completed(futures):
                variant, n_feats, m = fut.result()
                row = {"system": system, "experiment": "ablation",
                       "variant": variant, "n_features": n_feats, **m}
                all_rows.append(row)
                save_results(all_rows, out_dir, "ablation_results.csv")
                print(f"    Variant {variant:15s} ({n_feats} feats):  MCC={m['mcc']:.4f}  BalAcc={m['balanced_acc']:.4f}", flush=True)

    df = pd.DataFrame(all_rows)
    full_rf = [r for r in all_rows if r.get("variant")=="full" and r.get("model")=="RandomForest"]
    if full_rf:
        ref_mcc = float(full_rf[0]["mcc"])
        print(f"\n  MCC drops vs. full model (RF, {system}):  ref={ref_mcc:.4f}", flush=True)
        for _, row in df[df["model"]=="RandomForest"].iterrows():
            drop = ref_mcc - float(row["mcc"])
            print(f"    {row['variant']:15s}  MCC={float(row['mcc']):.4f}  drop={drop:+.4f}", flush=True)
    return df


# ==============================================================================
# EXPERIMENT 3: SEED VARIABILITY (R3-C2) -- STRATEGY 2 + 3 PARALLEL
# ==============================================================================
def run_seed_variability(system, stream_lite, out_dir, max_workers=3, worker_threads=2):
    print(f"\n{'='*72}", flush=True)
    print(f"  EXPERIMENT 3: SEED VARIABILITY (Strategy 2+3 Parallel, 10 seeds) [{system}]", flush=True)
    print(f"{'='*72}", flush=True)

    seed_file = out_dir / "seed_variability_results.csv"
    all_rows = []
    if seed_file.exists():
        try:
            all_rows = pd.read_csv(seed_file).to_dict("records")
            print(f"  Loaded {len(all_rows)} existing seed rows from {seed_file}", flush=True)
        except Exception:
            all_rows = []

    done_seeds = {r["seed"] for r in all_rows if "seed" in r}

    # Ensure seed 42 is loaded from baselines if available
    if 42 not in done_seeds:
        base_file = out_dir / "baselines_results.csv"
        if base_file.exists():
            bdf = pd.read_csv(base_file)
            rf_full = bdf[bdf["model"]=="RandomForest"]
            if len(rf_full) > 0:
                frow = rf_full.iloc[0].to_dict()
                frow["experiment"] = "seed_variability"
                frow["seed"] = 42
                all_rows.append(frow)
                done_seeds.add(42)
                save_results(all_rows, out_dir, "seed_variability_results.csv")
                print(f"  -- Seed 42 [Loaded from baselines: MCC={float(frow.get('mcc',0)):.4f}] --", flush=True)

    pending_seeds = [s for s in SEEDS if s not in done_seeds]
    if not pending_seeds:
        print(f"  -- All 10 seeds already completed --", flush=True)
    else:
        print(f"  Evaluating {len(pending_seeds)} seeds with {max_workers} parallel workers...", flush=True)
        tasks = [(s, stream_lite, ALL_FEATURE_COLS, worker_threads) for s in pending_seeds]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker_eval_seed, t): t[0] for t in tasks}
            for fut in concurrent.futures.as_completed(futures):
                seed, m = fut.result()
                row = {"system": system, "experiment": "seed_variability", "seed": seed, **m}
                all_rows.append(row)
                save_results(all_rows, out_dir, "seed_variability_results.csv")
                print(f"    Seed {seed:3d}:  MCC={m['mcc']:.4f}  BalAcc={m['balanced_acc']:.4f}", flush=True)

    df = pd.DataFrame(all_rows)
    print(f"\n  Summary (mean +/- std over {len(SEEDS)} seeds) -- {system}:", flush=True)
    for model_name, grp in df.groupby("model"):
        mcc_mean, mcc_std = grp["mcc"].mean(), grp["mcc"].std()
        ba_mean,  ba_std  = grp["balanced_acc"].mean(), grp["balanced_acc"].std()
        auc_mean, auc_std = grp["auc_roc"].mean(), grp["auc_roc"].std()
        print(f"    {model_name:20s}  MCC={mcc_mean:.4f}+/-{mcc_std:.4f}  "
              f"BalAcc={ba_mean:.4f}+/-{ba_std:.4f}  AUC={auc_mean:.4f}+/-{auc_std:.4f}", flush=True)
    return df


# ==============================================================================
# EXPERIMENT 4: PARAMETER SENSITIVITY (R1-C6) -- STRATEGY 2 + 3 PARALLEL
# ==============================================================================
def run_param_sensitivity(system, stream_lite, out_dir, max_workers=3, worker_threads=2):
    print(f"\n{'='*72}", flush=True)
    print(f"  EXPERIMENT 4: PARAMETER SENSITIVITY (Strategy 2+3 Parallel) [{system}]", flush=True)
    print(f"{'='*72}", flush=True)

    param_file = out_dir / "param_sensitivity_results.csv"
    all_rows = []
    if param_file.exists():
        try:
            all_rows = pd.read_csv(param_file).to_dict("records")
            print(f"  Loaded {len(all_rows)} existing param rows from {param_file}", flush=True)
        except Exception:
            all_rows = []

    done_configs = {r["config"] for r in all_rows if "config" in r}

    # Ensure default is loaded from baselines if available
    if "default" not in done_configs:
        base_file = out_dir / "baselines_results.csv"
        if base_file.exists():
            bdf = pd.read_csv(base_file)
            rf_full = bdf[bdf["model"]=="RandomForest"]
            if len(rf_full) > 0:
                frow = rf_full.iloc[0].to_dict()
                frow["experiment"] = "param_sensitivity"
                frow["config"] = "default"
                frow["warmup_revs"] = 5
                frow["min_train"] = 30
                frow["calib_window"] = 50
                all_rows.append(frow)
                done_configs.add("default")
                save_results(all_rows, out_dir, "param_sensitivity_results.csv")
                print(f"  -- Config: default [Loaded from baselines: MCC={float(frow.get('mcc',0)):.4f}] --", flush=True)

    pending_configs = [c for c in PARAM_GRID if c[3] not in done_configs]
    if not pending_configs:
        print(f"  -- All parameter configs already completed --", flush=True)
    else:
        print(f"  Evaluating {len(pending_configs)} configs with {max_workers} parallel workers...", flush=True)
        tasks = [(c, stream_lite, ALL_FEATURE_COLS, 42, worker_threads) for c in pending_configs]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker_eval_param, t): t[0][3] for t in tasks}
            for fut in concurrent.futures.as_completed(futures):
                config_tuple, m = fut.result()
                warmup, min_train, calib, label = config_tuple
                row = {"system": system, "experiment": "param_sensitivity",
                       "config": label, "warmup_revs": warmup,
                       "min_train": min_train, "calib_window": calib, **m}
                all_rows.append(row)
                save_results(all_rows, out_dir, "param_sensitivity_results.csv")
                print(f"    Config {label:18s}:  MCC={m['mcc']:.4f}  BalAcc={m['balanced_acc']:.4f}", flush=True)

    df = pd.DataFrame(all_rows)
    ref = df[(df["model"]=="RandomForest") & (df["config"]=="default")]["mcc"].values
    if len(ref):
        print(f"\n  MCC vs. default (RF, {system}):  default={ref[0]:.4f}", flush=True)
        for _, row in df[df["model"]=="RandomForest"].iterrows():
            diff = float(row["mcc"]) - float(ref[0])
            print(f"    {row['config']:18s}  MCC={float(row['mcc']):.4f}  diff={diff:+.4f}", flush=True)
    return df


# ==============================================================================
# EXPERIMENT 5: GROUND-TRUTH SENSITIVITY (R1-C4 / R3-C6)
# ==============================================================================
def run_ground_truth(system, data, stream_lite, out_dir, commit_log_path=None):
    print(f"\n{'='*72}", flush=True)
    print(f"  EXPERIMENT 5: GROUND-TRUTH SENSITIVITY  [{system}]", flush=True)
    print(f"{'='*72}", flush=True)

    gt_file = out_dir / "ground_truth_results.csv"
    auto_revs = load_automated_revisions(commit_log_path)
    expected_rows = 3 if auto_revs else 2

    all_rows = []
    done_modes = set()
    if gt_file.exists():
        try:
            gdf = pd.read_csv(gt_file)
            for _, r in gdf.iterrows():
                done_modes.add(r["label_mode"])
                all_rows.append(r.to_dict())
            if len(done_modes) >= expected_rows:
                print(f"  -- Ground truth sensitivity already completed in {gt_file}, skipping --", flush=True)
                return gdf
        except Exception:
            pass

    frag_idx, pair_idx, frag_revs, pair_revs, change_revs = data

    # 1. Default (already in stream_lite or baselines)
    if "default" not in done_modes:
        print(f"\n  -- Label mode: default --", flush=True)
        base_file = out_dir / "baselines_results.csv"
        loaded_def = False
        if base_file.exists():
            try:
                bdf = pd.read_csv(base_file)
                rf_row = bdf[bdf["model"]=="RandomForest"]
                if len(rf_row):
                    m = rf_row.iloc[0].to_dict()
                    row = {"system": system, "experiment": "ground_truth", "label_mode": "default", **m}
                    all_rows.append(row)
                    done_modes.add("default")
                    save_results(all_rows, out_dir, "ground_truth_results.csv")
                    print(f"    {'RandomForest':20s}  MCC={float(m.get('mcc',0)):.4f}  BalAcc={float(m.get('balanced_acc',0)):.4f} [Loaded from baselines]", flush=True)
                    loaded_def = True
            except Exception:
                pass
        if not loaded_def:
            models = get_models(seed=42, best_only=True, n_jobs=-1)
            res_def = walk_forward(
                None, None, None, None, None,
                feature_cols=ALL_FEATURE_COLS,
                models=models, baselines={},
                warmup_revs=5, min_train=30, calib_window=50,
                seed=42, verbose=False, stream=stream_lite,
            )
            for name, m in res_def.items():
                all_rows.append({"system": system, "experiment": "ground_truth", "label_mode": "default", **m})
                print(f"    {name:20s}  MCC={m['mcc']:.4f}  BalAcc={m['balanced_acc']:.4f}  n={m['n']}", flush=True)
            done_modes.add("default")
            save_results(all_rows, out_dir, "ground_truth_results.csv")

    # 2. Strict Module (evaluate with actual_label_strict_module)
    if "strict_module" not in done_modes:
        print(f"\n  -- Label mode: strict_module --", flush=True)
        # Check if strict_module labels differ from actual_label
        diff_count = 0
        same_count = 0
        for R in change_revs:
            alive = frag_idx.get(R)
            if alive is None: continue
            changed = alive[alive["changeType"].str.upper() == "M"]
            for _, r in changed.iterrows():
                g = int(r["gcid"])
                y1 = actual_label(g, R, pair_idx, frag_idx)
                y2 = actual_label_strict_module(g, R, pair_idx, frag_idx)
                if y1 == y2:
                    same_count += 1
                else:
                    diff_count += 1

        if diff_count == 0 and "default" in done_modes:
            print(f"    [Strategy 2] All {same_count} labels match default -- reusing default metrics", flush=True)
            def_rows = [r for r in all_rows if r.get("label_mode") == "default"]
            if def_rows:
                s_row = def_rows[0].copy()
                s_row["label_mode"] = "strict_module"
                all_rows.append(s_row)
                done_modes.add("strict_module")
                save_results(all_rows, out_dir, "ground_truth_results.csv")
                print(f"    {'RandomForest':20s}  MCC={float(s_row.get('mcc',0)):.4f}  BalAcc={float(s_row.get('balanced_acc',0)):.4f} [Strict=Default]", flush=True)
        else:
            models = get_models(seed=42, best_only=True, n_jobs=-1)
            res_strict = walk_forward(
                frag_idx, pair_idx, frag_revs, pair_revs, change_revs,
                feature_cols=ALL_FEATURE_COLS,
                label_fn=actual_label_strict_module,
                models=models, baselines={},
                warmup_revs=5, min_train=30, calib_window=50,
                seed=42, verbose=False,
            )
            for name, m in res_strict.items():
                all_rows.append({"system": system, "experiment": "ground_truth", "label_mode": "strict_module", **m})
                print(f"    {name:20s}  MCC={m['mcc']:.4f}  BalAcc={m['balanced_acc']:.4f}  n={m['n']}", flush=True)
            done_modes.add("strict_module")
            save_results(all_rows, out_dir, "ground_truth_results.csv")

    # 3. Excluded Automated Revisions (if commit log available)
    if auto_revs and "excl_automated" not in done_modes:
        print(f"\n  -- Label mode: excl_automated ({len(auto_revs)} excluded revs) --", flush=True)
        # Filter stream_lite to omit auto_revs
        stream_excl = [item for item in stream_lite if item[0] not in auto_revs]
        models = get_models(seed=42, best_only=True, n_jobs=-1)
        res_excl = walk_forward(
            None, None, None, None, None,
            feature_cols=ALL_FEATURE_COLS,
            models=models, baselines={},
            warmup_revs=5, min_train=30, calib_window=50,
            seed=42, verbose=False, stream=stream_excl,
        )
        for name, m in res_excl.items():
            all_rows.append({"system": system, "experiment": "ground_truth", "label_mode": "excl_automated", **m})
            print(f"    {name:20s}  MCC={m['mcc']:.4f}  BalAcc={m['balanced_acc']:.4f}  n={m['n']}", flush=True)
        done_modes.add("excl_automated")
        save_results(all_rows, out_dir, "ground_truth_results.csv")

    df = save_results(all_rows, out_dir, "ground_truth_results.csv")
    return df


# ==============================================================================
# MASTER RUNNER
# ==============================================================================
def run_system_accelerated(system_name, system_cfg, experiments, base_out, max_workers=3, worker_threads=2):
    frag_path   = system_cfg["frag"]
    pair_path   = system_cfg["pair"]
    commit_log  = system_cfg.get("commit_log")
    out_dir     = base_out / system_name

    print(f"\n{'#'*72}", flush=True)
    print(f"#  ACCELERATED SYSTEM EVALUATION: {system_name}", flush=True)
    print(f"#  Workers: {max_workers} | Threads per worker: {worker_threads}", flush=True)
    print(f"{'#'*72}", flush=True)

    frag_df, pair_df, frag_idx, pair_idx, frag_revs, pair_revs, change_revs = \
        load_data(str(frag_path), str(pair_path))
    data = (frag_idx, pair_idx, frag_revs, pair_revs, change_revs)

    t_sys = time.time()

    # Step 1: Baselines (if needed)
    if "baselines" in experiments:
        run_baselines(system_name, data, out_dir)

    # Step 2: Precomputed Stream (Strategy 2)
    stream_lite = None
    if any(e in experiments for e in ("ablation", "seeds", "params", "gt")):
        stream_lite = get_or_create_stream(system_name, data, out_dir)

    # Step 3: Parallel Evaluations (Strategy 3)
    if "ablation" in experiments:
        run_ablation(system_name, stream_lite, out_dir, max_workers=max_workers, worker_threads=worker_threads)

    if "seeds" in experiments:
        run_seed_variability(system_name, stream_lite, out_dir, max_workers=max_workers, worker_threads=worker_threads)

    if "params" in experiments:
        run_param_sensitivity(system_name, stream_lite, out_dir, max_workers=max_workers, worker_threads=worker_threads)

    if "gt" in experiments:
        run_ground_truth(system_name, data, stream_lite, out_dir, commit_log)

    print(f"\n  [OK] {system_name} accelerated run completed in {(time.time()-t_sys)/60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser(description="Accelerated Multi-Core Reviewer Response Suite")
    p.add_argument("--systems", nargs="+", default=["dnsjava", "Jmol"],
                   help="Systems to run (tuxguitar, Ctags, dnsjava, Jmol)")
    p.add_argument("--experiments", nargs="+",
                   default=["baselines", "ablation", "seeds", "params", "gt"],
                   help="Experiments: baselines ablation seeds params gt")
    p.add_argument("--workers", type=int, default=3, help="Max parallel process workers (Strategy 3)")
    p.add_argument("--threads", type=int, default=2, help="Threads per worker model fit")
    p.add_argument("--output", default=None, help="Output directory")
    args = p.parse_args()

    base_out = Path(args.output) if args.output else \
               Path(__file__).resolve().parent / "results" / "experiments"
    base_out.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    for sname in args.systems:
        if sname not in SYSTEMS:
            print(f"[WARN] Unknown system: {sname}. Available: {list(SYSTEMS)}", flush=True)
            continue
        run_system_accelerated(sname, SYSTEMS[sname], set(args.experiments), base_out,
                               max_workers=args.workers, worker_threads=args.threads)

    print(f"\n{'='*72}", flush=True)
    print(f"  ALL ACCELERATED RUNS COMPLETED in {(time.time()-t_total)/60:.1f} minutes", flush=True)
    print(f"  Results saved to: {base_out}", flush=True)
    print(f"{'='*72}", flush=True)


if __name__ == "__main__":
    main()
