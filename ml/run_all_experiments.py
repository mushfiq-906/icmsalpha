"""
run_all_experiments.py
======================
Master runner that executes ALL reviewer-requested experiments for
tuxguitar and Ctags (or any systems passed on the command line).

Experiments:
  1. BASELINES       -- WCS-threshold + co-change-rate heuristic baselines (R1-C2/R3-C1)
  2. ABLATION        -- Leave-one-group-out feature ablation            (R1-C3)
  3. SEED_VARIABILITY -- 10 random seeds, report mean +/- std             (R3-C2)
  4. PARAM_SWEEP     -- Sweep WARMUP_REVS / MIN_TRAIN / CALIB_WINDOW   (R1-C6)
  5. GROUND_TRUTH    -- Strict-module + excl-automated label variants   (R1-C4/R3-C6)

Results are written to:
  ml/results/experiments/<system>/<experiment>_results.csv
"""
import sys, os, re, time, warnings, argparse

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
from pathlib import Path
from sklearn.metrics import matthews_corrcoef

from experiment_utils import (
    load_data, walk_forward, get_models, compute_metrics,
    actual_label, actual_label_strict_module, make_excl_auto_label_fn,
    wcs_threshold_predict, cochange_rate_predict,
    ALL_FEATURE_COLS, ABLATION_VARIANTS,
)

# -- SYSTEM PATHS --------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent / "WorkFolder"
NICAD_BASE = Path(r"C:\Users\CSE-AI-Lab\Desktop\Thesis\Nicad\systems")

SYSTEMS = {
    "tuxguitar": {
        "frag": BASE / "tuxguitar" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_fragment.csv",
        "pair": BASE / "tuxguitar" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_pair.csv",
        "commit_log": None,   # GitHub project -- no SVN log
    },
    "Ctags": {
        "frag": BASE / "Ctags" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_fragment.csv",
        "pair": BASE / "Ctags" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_pair.csv",
        "commit_log": None,
    },
    "jEdit": {
        "frag": BASE / "jEdit" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_fragment.csv",
        "pair": BASE / "jEdit" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_pair.csv",
        "commit_log": NICAD_BASE / "jEdit" / "commit_logs.txt",
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
    "jmol": {
        "frag": BASE / "Jmol" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_fragment.csv",
        "pair": BASE / "Jmol" / "Datasets" / "CloneGenealogy" / "Type3_Block_rev_pair.csv",
        "commit_log": NICAD_BASE / "Jmol" / "commit_logs.txt",
    },
}

# -- COMMIT LOG PARSING --------------------------------------------------------
AUTO_PATTERNS = re.compile(
    r'\breformat\b|\bformatting\b|\bindent(ation)?\b|\bwhitespace\b|'
    r'\bmerge\b|\bbump.version\b|\brelease\b|\bupdate.version\b|'
    r'^\*\*\*.empty.log|\binitial.revision\b|\bimport.of\b|'
    r'\bauto.?generated\b|\bno.message\b',
    re.IGNORECASE
)

def load_automated_revisions(commit_log_path):
    """Parse jEdit-style commit log and return set of automated revision numbers."""
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


# -- OUTPUT HELPERS ------------------------------------------------------------
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
# EXPERIMENT 1: BASELINES  (R1-C2 / R3-C1)
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
        "WCS-Threshold":    wcs_threshold_predict,
        "CoChange-Rate":    cochange_rate_predict,
    }
    models = get_models(seed=42, best_only=True)

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

    # Merge existing verified metrics for other models if available
    try:
        hist_path = BASE.parent / "ml" / "results" / "predict_standing" / system / f"standing_{system}_Type3_Block_global_metrics.csv"
        if hist_path.exists():
            hist_df = pd.read_csv(hist_path)
            existing_models = {r["model"] for r in rows}
            for _, hrow in hist_df.iterrows():
                mn = hrow["model"]
                if mn not in existing_models:
                    rows.append({"system": system, "experiment": "baselines", **hrow.to_dict()})
    except Exception as e:
        pass

    save_results(rows, out_dir, "baselines_results.csv")
    print_results(results, f"Baselines -- {system}")
    return results


# ==============================================================================
# EXPERIMENT 2: FEATURE ABLATION  (R1-C3)
# ==============================================================================
def run_ablation(system, data, out_dir):
    print(f"\n{'='*72}", flush=True)
    print(f"  EXPERIMENT 2: FEATURE ABLATION  [{system}]", flush=True)
    print(f"{'='*72}", flush=True)
    frag_idx, pair_idx, frag_revs, pair_revs, change_revs = data

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

    for variant, feature_cols in ABLATION_VARIANTS.items():
        if variant in done_variants:
            print(f"  -- Variant: {variant} already completed, skipping --", flush=True)
            continue

        print(f"\n  -- Variant: {variant} ({len(feature_cols)} features) --", flush=True)
        models  = get_models(seed=42, best_only=True)   # RF only for speed & stability
        results = walk_forward(
            frag_idx, pair_idx, frag_revs, pair_revs, change_revs,
            feature_cols=feature_cols,
            label_fn=actual_label,
            models=models,
            baselines={},
            warmup_revs=5, min_train=30, calib_window=50,
            seed=42, verbose=False,
        )
        for name, m in results.items():
            row = {"system": system, "experiment": "ablation",
                   "variant": variant, "n_features": len(feature_cols), **m}
            all_rows.append(row)
        save_results(all_rows, out_dir, "ablation_results.csv")
        print_results(results, f"Ablation [{variant}] -- {system}")

    df = pd.DataFrame(all_rows)
    # Print MCC drops
    full_rf = [r for r in all_rows if r.get("variant")=="full" and r.get("model")=="RandomForest"]
    if full_rf:
        ref_mcc = float(full_rf[0]["mcc"])
        print(f"\n  MCC drops vs. full model (RF, {system}):  ref={ref_mcc:.4f}", flush=True)
        for _, row in df[df["model"]=="RandomForest"].iterrows():
            drop = ref_mcc - float(row["mcc"])
            print(f"    {row['variant']:15s}  MCC={float(row['mcc']):.4f}  drop={drop:+.4f}", flush=True)
    return df


# ==============================================================================
# EXPERIMENT 3: SEED VARIABILITY  (R3-C2)
# ==============================================================================
SEEDS = [42, 0, 1, 7, 13, 17, 21, 37, 99, 123]

def run_seed_variability(system, data, out_dir):
    print(f"\n{'='*72}", flush=True)
    print(f"  EXPERIMENT 3: SEED VARIABILITY (10 seeds)  [{system}]", flush=True)
    print(f"{'='*72}", flush=True)
    frag_idx, pair_idx, frag_revs, pair_revs, change_revs = data

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

    for seed in SEEDS:
        if seed in done_seeds:
            print(f"  -- Seed {seed} already completed, skipping --", flush=True)
            continue

        print(f"\n  -- Seed {seed} --", flush=True)
        models  = get_models(seed=seed, best_only=True)
        results = walk_forward(
            frag_idx, pair_idx, frag_revs, pair_revs, change_revs,
            feature_cols=ALL_FEATURE_COLS,
            label_fn=actual_label,
            models=models, baselines={},
            warmup_revs=5, min_train=30, calib_window=50,
            seed=seed, verbose=False,
        )
        for name, m in results.items():
            row = {"system": system, "experiment": "seed_variability", "seed": seed, **m}
            all_rows.append(row)
            print(f"    {name:20s}  MCC={m['mcc']:.4f}  BalAcc={m['balanced_acc']:.4f}", flush=True)
        save_results(all_rows, out_dir, "seed_variability_results.csv")

    df = pd.DataFrame(all_rows)
    # Print mean +/- std summary
    print(f"\n  Summary (mean +/- std over {len(SEEDS)} seeds) -- {system}:", flush=True)
    for model_name, grp in df.groupby("model"):
        mcc_mean = grp["mcc"].mean(); mcc_std = grp["mcc"].std()
        ba_mean  = grp["balanced_acc"].mean(); ba_std = grp["balanced_acc"].std()
        auc_mean = grp["auc_roc"].mean(); auc_std = grp["auc_roc"].std()
        print(f"    {model_name:20s}  MCC={mcc_mean:.4f}+/-{mcc_std:.4f}  "
              f"BalAcc={ba_mean:.4f}+/-{ba_std:.4f}  AUC={auc_mean:.4f}+/-{auc_std:.4f}", flush=True)
    return df


# ==============================================================================
# EXPERIMENT 4: PARAMETER SENSITIVITY  (R1-C6)
# ==============================================================================
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

def run_param_sensitivity(system, data, out_dir):
    print(f"\n{'='*72}", flush=True)
    print(f"  EXPERIMENT 4: PARAMETER SENSITIVITY  [{system}]", flush=True)
    print(f"{'='*72}", flush=True)
    frag_idx, pair_idx, frag_revs, pair_revs, change_revs = data

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

    for warmup, min_train, calib, label in PARAM_GRID:
        if label in done_configs:
            print(f"  -- Config: {label} already completed, skipping --", flush=True)
            continue

        print(f"\n  -- Config: {label} --", flush=True)
        models  = get_models(seed=42, best_only=True)
        results = walk_forward(
            frag_idx, pair_idx, frag_revs, pair_revs, change_revs,
            feature_cols=ALL_FEATURE_COLS,
            label_fn=actual_label,
            models=models, baselines={},
            warmup_revs=warmup, min_train=min_train, calib_window=min(calib, len(change_revs)),
            seed=42, verbose=False,
        )
        for name, m in results.items():
            row = {"system": system, "experiment": "param_sensitivity",
                   "config": label, "warmup_revs": warmup,
                   "min_train": min_train, "calib_window": calib, **m}
            all_rows.append(row)
            print(f"    {name:20s}  MCC={m['mcc']:.4f}  BalAcc={m['balanced_acc']:.4f}", flush=True)
        save_results(all_rows, out_dir, "param_sensitivity_results.csv")

    df = pd.DataFrame(all_rows)
    ref = df[(df["model"]=="RandomForest") & (df["config"]=="default")]["mcc"].values
    if len(ref):
        print(f"\n  MCC vs. default (RF, {system}):  default={ref[0]:.4f}", flush=True)
        for _, row in df[df["model"]=="RandomForest"].iterrows():
            diff = float(row["mcc"]) - float(ref[0])
            print(f"    {row['config']:18s}  MCC={float(row['mcc']):.4f}  diff={diff:+.4f}", flush=True)
    return df


# ==============================================================================
# EXPERIMENT 5: GROUND-TRUTH SENSITIVITY  (R1-C4 / R3-C6)
# ==============================================================================
def run_ground_truth(system, data, out_dir, commit_log_path=None):
    print(f"\n{'='*72}", flush=True)
    print(f"  EXPERIMENT 5: GROUND-TRUTH SENSITIVITY  [{system}]", flush=True)
    print(f"{'='*72}", flush=True)
    frag_idx, pair_idx, frag_revs, pair_revs, change_revs = data

    auto_revs = load_automated_revisions(commit_log_path)

    labeling_modes = {
        "default":     actual_label,
        "strict_module": actual_label_strict_module,
    }
    if auto_revs:
        labeling_modes["excl_automated"] = make_excl_auto_label_fn(auto_revs)

    all_rows = []
    for mode_name, label_fn in labeling_modes.items():
        print(f"\n  -- Label mode: {mode_name} --", flush=True)
        models  = get_models(seed=42, best_only=True)
        results = walk_forward(
            frag_idx, pair_idx, frag_revs, pair_revs, change_revs,
            feature_cols=ALL_FEATURE_COLS,
            label_fn=label_fn,
            models=models, baselines={},
            warmup_revs=5, min_train=30, calib_window=50,
            seed=42, verbose=False,
        )
        for name, m in results.items():
            row = {"system": system, "experiment": "ground_truth",
                   "label_mode": mode_name, **m}
            all_rows.append(row)
            print(f"    {name:20s}  MCC={m['mcc']:.4f}  "
                  f"BalAcc={m['balanced_acc']:.4f}  n={m['n']}", flush=True)
    print_results({r["label_mode"]+"/"+r["model"]: r for r in all_rows if r["model"]=="RandomForest"},
                  f"Ground-Truth Sensitivity -- {system}")
    df = save_results(all_rows, out_dir, "ground_truth_results.csv")
    return df


# ==============================================================================
# MASTER RUNNER
# ==============================================================================
def run_system(system_name, system_cfg, experiments, base_out):
    frag_path      = system_cfg["frag"]
    pair_path      = system_cfg["pair"]
    commit_log     = system_cfg.get("commit_log")
    out_dir        = base_out / system_name

    print(f"\n{'#'*72}", flush=True)
    print(f"#  SYSTEM: {system_name}", flush=True)
    print(f"{'#'*72}", flush=True)

    # Load data once for all experiments
    frag_df, pair_df, frag_idx, pair_idx, frag_revs, pair_revs, change_revs = \
        load_data(str(frag_path), str(pair_path))
    data = (frag_idx, pair_idx, frag_revs, pair_revs, change_revs)

    t_sys = time.time()
    if "baselines"  in experiments: run_baselines(system_name, data, out_dir)
    if "ablation"   in experiments: run_ablation(system_name, data, out_dir)
    if "seeds"      in experiments: run_seed_variability(system_name, data, out_dir)
    if "params"     in experiments: run_param_sensitivity(system_name, data, out_dir)
    if "gt"         in experiments: run_ground_truth(system_name, data, out_dir, commit_log)

    print(f"\n  [OK] {system_name} done in {(time.time()-t_sys)/60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser(description="Run all reviewer-response experiments")
    p.add_argument("--systems",     nargs="+", default=["tuxguitar","Ctags"],
                   help="Systems to run (tuxguitar, Ctags, jEdit)")
    p.add_argument("--experiments", nargs="+",
                   default=["baselines","ablation","seeds","params","gt"],
                   help="Experiments: baselines ablation seeds params gt")
    p.add_argument("--output", default=None, help="Output base dir")
    args = p.parse_args()

    base_out = Path(args.output) if args.output else \
               Path(__file__).resolve().parent / "results" / "experiments"
    base_out.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    for sname in args.systems:
        if sname not in SYSTEMS:
            print(f"[WARN] Unknown system: {sname}. Available: {list(SYSTEMS)}", flush=True)
            continue
        run_system(sname, SYSTEMS[sname], set(args.experiments), base_out)

    print(f"\n{'='*72}", flush=True)
    print(f"  ALL DONE in {(time.time()-t_total)/60:.1f} minutes", flush=True)
    print(f"  Results: {base_out}", flush=True)
    print(f"{'='*72}", flush=True)


if __name__ == "__main__":
    main()
