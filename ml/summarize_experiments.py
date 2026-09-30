"""
summarize_experiments.py
========================
Reads all experiment CSV outputs and prints paper-ready tables.
Run after run_all_experiments.py finishes.

Usage:
    python summarize_experiments.py --systems tuxguitar Ctags
"""
import sys, argparse
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd
import numpy as np
from pathlib import Path

BASE_OUT = Path(__file__).resolve().parent / "results" / "experiments"


def fmt(val, decimals=4):
    try: return f"{float(val):.{decimals}f}"
    except: return str(val)


def load(system, filename):
    path = BASE_OUT / system / filename
    if not path.exists():
        print(f"  [MISSING] {path}", flush=True)
        return None
    return pd.read_csv(path)


# -- TABLE 1: BASELINES --------------------------------------------------------
def print_baselines_table(systems):
    print("\n" + "="*80, flush=True)
    print("TABLE: Baselines vs. ML models (R1-C2 / R3-C1)", flush=True)
    print("="*80, flush=True)
    hdr = f"{'Model':22s} {'System':12s} {'MCC':>8} {'BalAcc':>8} {'AUC':>8} {'F1':>8} {'n':>8}"
    print(hdr, flush=True)
    print("-"*80, flush=True)
    for sys in systems:
        df = load(sys, "baselines_results.csv")
        if df is None: continue
        for _, row in df.iterrows():
            tag = "[BASELINE]" if row["model"] in ["WCS-Threshold","CoChange-Rate"] else ""
            print(f"  {row['model']:20s}{tag:3s} {sys:12s} "
                  f"{fmt(row['mcc']):>8} {fmt(row['balanced_acc']):>8} "
                  f"{fmt(row['auc_roc']):>8} {fmt(row['f1']):>8} {int(row['n']):>8}", flush=True)
    print(flush=True)


# -- TABLE 2: ABLATION ---------------------------------------------------------
def print_ablation_table(systems):
    print("\n" + "="*80, flush=True)
    print("TABLE: Feature group ablation -- MCC drop vs. full model (R1-C3)", flush=True)
    print("="*80, flush=True)
    hdr = f"{'Variant':18s} {'Features':>8} {'System':12s} {'MCC':>8} {'Drop vs. full':>14}"
    print(hdr, flush=True)
    print("-"*80, flush=True)
    for sys in systems:
        df = load(sys, "ablation_results.csv")
        if df is None: continue
        rf_df = df[df["model"] == "RandomForest"]
        ref_row = rf_df[rf_df["variant"] == "full"]
        if len(ref_row) == 0: continue
        ref_mcc = float(ref_row.iloc[0]["mcc"])
        for _, row in rf_df.iterrows():
            drop = row["mcc"] - ref_mcc
            star = "  <-" if abs(drop) > 0.05 else ""
            print(f"  {row['variant']:18s} {int(row['n_features']):>8} {sys:12s} "
                  f"{fmt(row['mcc']):>8} {drop:>+14.4f}{star}", flush=True)
    print(flush=True)


# -- TABLE 3: SEED VARIABILITY -------------------------------------------------
def print_seed_table(systems):
    print("\n" + "="*80, flush=True)
    print("TABLE: Seed variability -- mean +/- std over 10 seeds (R3-C2)", flush=True)
    print("="*80, flush=True)
    hdr = f"{'Model':22s} {'System':12s} {'MCC mean+/-std':>16} {'BalAcc mean+/-std':>18} {'AUC mean+/-std':>16}"
    print(hdr, flush=True)
    print("-"*80, flush=True)
    for sys in systems:
        df = load(sys, "seed_variability_results.csv")
        if df is None: continue
        for model, grp in df.groupby("model"):
            mcc_m, mcc_s = grp["mcc"].mean(), grp["mcc"].std()
            ba_m,  ba_s  = grp["balanced_acc"].mean(), grp["balanced_acc"].std()
            auc_m, auc_s = grp["auc_roc"].mean(), grp["auc_roc"].std()
            print(f"  {model:22s} {sys:12s} "
                  f"{mcc_m:.4f}+/-{mcc_s:.4f}  "
                  f"{ba_m:.4f}+/-{ba_s:.4f}    "
                  f"{auc_m:.4f}+/-{auc_s:.4f}", flush=True)
    print(flush=True)


# -- TABLE 4: PARAMETER SENSITIVITY -------------------------------------------
def print_param_table(systems):
    print("\n" + "="*80, flush=True)
    print("TABLE: Parameter sensitivity -- MCC vs. default (R1-C6)", flush=True)
    print("="*80, flush=True)
    hdr = f"{'Config':20s} {'System':12s} {'MCC':>8} {'vs. default':>12}"
    print(hdr, flush=True)
    print("-"*80, flush=True)
    for sys in systems:
        df = load(sys, "param_sensitivity_results.csv")
        if df is None: continue
        rf = df[df["model"]=="RandomForest"]
        ref_rows = rf[rf["config"]=="default"]
        if len(ref_rows) == 0: continue
        ref_mcc = float(ref_rows.iloc[0]["mcc"])
        for _, row in rf.iterrows():
            diff = row["mcc"] - ref_mcc
            star = "  *" if abs(diff) > 0.02 else ""
            print(f"  {row['config']:20s} {sys:12s} {fmt(row['mcc']):>8} {diff:>+12.4f}{star}", flush=True)
    print(flush=True)


# -- TABLE 5: GROUND-TRUTH SENSITIVITY ----------------------------------------
def print_gt_table(systems):
    print("\n" + "="*80, flush=True)
    print("TABLE: Ground-truth sensitivity -- label mode comparison (R1-C4/R3-C6)", flush=True)
    print("="*80, flush=True)
    hdr = f"{'Label mode':22s} {'System':12s} {'MCC':>8} {'BalAcc':>8} {'n':>8} {'vs. default':>12}"
    print(hdr, flush=True)
    print("-"*80, flush=True)
    for sys in systems:
        df = load(sys, "ground_truth_results.csv")
        if df is None: continue
        rf = df[df["model"]=="RandomForest"]
        ref_rows = rf[rf["label_mode"]=="default"]
        ref_mcc  = float(ref_rows.iloc[0]["mcc"]) if len(ref_rows) else None
        for _, row in rf.iterrows():
            diff = (row["mcc"] - ref_mcc) if ref_mcc else 0.0
            print(f"  {row['label_mode']:22s} {sys:12s} "
                  f"{fmt(row['mcc']):>8} {fmt(row['balanced_acc']):>8} "
                  f"{int(row['n']):>8} {diff:>+12.4f}", flush=True)
    print(flush=True)


def print_latex_tables(systems):
    print("\n" + "="*80, flush=True)
    print("LATEX TABLES FOR PAPER & REBUTTAL", flush=True)
    print("="*80, flush=True)

    # 1. Baselines Table
    print("\n% --- Table 1: Baselines ---")
    print(r"""\begin{table}[ht]
\centering
\caption{Performance Comparison: Coupling Heuristics vs. Walk-Forward ML Models}
\label{tab:baselines}
\small
\begin{tabular}{llccccc}
\toprule
Model & System & MCC & Bal.\ Acc & AUC-ROC & F1 & $N$ \\
\midrule""")
    for sys in systems:
        df = load(sys, "baselines_results.csv")
        if df is None: continue
        for _, row in df.iterrows():
            mname = row['model']
            if mname in ["WCS-Threshold", "CoChange-Rate"]:
                mname = r"\textbf{" + mname + r"} (Baseline)"
            print(f"{mname} & {sys} & {float(row['mcc']):.4f} & {float(row['balanced_acc']):.4f} & "
                  f"{float(row['auc_roc']):.4f} & {float(row['f1']):.4f} & {int(row['n'])} \\\\")
        print(r"\midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # 2. Ablation Table (Side-by-side)
    print("\n% --- Table 2: Feature Group Ablation ---")
    print(r"""\begin{table}[ht]
\centering
\caption{Feature Group Ablation: MCC and Degradation ($\Delta$MCC) vs.\ Full Model}
\label{tab:ablation}
\small
\begin{tabular}{lccccc}
\toprule
Configuration & $|F|$ & tuxguitar MCC & $\Delta$ & Ctags MCC & $\Delta$ \\
\midrule""")
    variants = [
        ("full", "Full Model (All Features)", 47),
        ("no_fragment", "w/o Fragment Stability", 38),
        ("no_process", "w/o Process Metrics", 42),
        ("no_pair", "w/o Historical Coupling", 43),
        ("no_decay", "w/o Temporal Decay", 41),
    ]
    dfs = {sys: load(sys, "ablation_results.csv") for sys in systems}
    for var_id, var_label, n_feat in variants:
        row_str = f"{var_label} & {n_feat}"
        for sys in systems:
            df = dfs.get(sys)
            if df is not None:
                rf_df = df[df["model"]=="RandomForest"]
                ref_row = rf_df[rf_df["variant"]=="full"]
                cur_row = rf_df[rf_df["variant"]==var_id]
                if len(ref_row) and len(cur_row):
                    ref_mcc = float(ref_row.iloc[0]["mcc"])
                    cur_mcc = float(cur_row.iloc[0]["mcc"])
                    diff = cur_mcc - ref_mcc
                    diff_str = "--" if var_id == "full" else f"{diff:+.4f}"
                    row_str += f" & {cur_mcc:.4f} & {diff_str}"
                else:
                    row_str += " & -- & --"
            else:
                row_str += " & -- & --"
        row_str += r" \\"
        print(row_str)
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # 3. Seed Variability Table
    print("\n% --- Table 3: Seed Variability ---")
    print(r"""\begin{table}[ht]
\centering
\caption{Model Stability across 10 Random Seeds (Random Forest)}
\label{tab:seed_variability}
\small
\begin{tabular}{lccc}
\toprule
System & MCC (mean $\pm$ std) & Bal.\ Acc (mean $\pm$ std) & AUC-ROC (mean $\pm$ std) \\
\midrule""")
    for sys in systems:
        df = load(sys, "seed_variability_results.csv")
        if df is not None:
            rf = df[df["model"]=="RandomForest"]
            if len(rf):
                mcc_m, mcc_s = rf["mcc"].mean(), rf["mcc"].std()
                ba_m,  ba_s  = rf["balanced_acc"].mean(), rf["balanced_acc"].std()
                auc_m, auc_s = rf["auc_roc"].mean(), rf["auc_roc"].std()
                print(f"{sys} & {mcc_m:.4f} $\\pm$ {mcc_s:.4f} & {ba_m:.4f} $\\pm$ {ba_s:.4f} & {auc_m:.4f} $\\pm$ {auc_s:.4f} \\\\")
            else:
                print(f"{sys} & Pending & Pending & Pending \\\\")
        else:
            print(f"{sys} & Pending & Pending & Pending \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # 4. Parameter Sensitivity Table
    print("\n% --- Table 4: Parameter Sensitivity ---")
    print(r"""\begin{table}[ht]
\centering
\caption{Hyperparameter Sensitivity under Walk-Forward Evaluation (Random Forest)}
\label{tab:param_sensitivity}
\small
\begin{tabular}{lcccc}
\toprule
Configuration & tuxguitar MCC & $\Delta$ & Ctags MCC & $\Delta$ \\
\midrule""")
    param_configs = [
        ("default", "Default ($W=5, M=30, C=50$)"),
        ("warmup=3", "Warmup $W=3$ revisions"),
        ("warmup=10", "Warmup $W=10$ revisions"),
        ("min_train=10", "Min Train $M=10$ events"),
        ("min_train=50", "Min Train $M=50$ events"),
        ("calib=20", "Calibration Window $C=20$"),
        ("calib=100", "Calibration Window $C=100$"),
        ("calib=inf", "Calibration Window $C=\\infty$ (Expanding)"),
    ]
    p_dfs = {sys: load(sys, "param_sensitivity_results.csv") for sys in systems}
    for cfg_id, cfg_label in param_configs:
        row_str = f"{cfg_label}"
        for sys in systems:
            df = p_dfs.get(sys)
            if df is not None:
                rf = df[df["model"]=="RandomForest"]
                ref_row = rf[rf["config"]=="default"]
                cur_row = rf[rf["config"]==cfg_id]
                if len(ref_row) and len(cur_row):
                    ref_mcc = float(ref_row.iloc[0]["mcc"])
                    cur_mcc = float(cur_row.iloc[0]["mcc"])
                    diff = cur_mcc - ref_mcc
                    diff_str = "--" if cfg_id == "default" else f"{diff:+.4f}"
                    row_str += f" & {cur_mcc:.4f} & {diff_str}"
                else:
                    row_str += " & -- & --"
            else:
                row_str += " & -- & --"
        row_str += r" \\"
        print(row_str)
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # 5. Ground-Truth Table
    print("\n% --- Table 5: Ground-Truth Sensitivity ---")
    print(r"""\begin{table}[ht]
\centering
\caption{Ground-Truth Sensitivity: Module-Constrained vs.\ Default Co-Change}
\label{tab:gt_sensitivity}
\small
\begin{tabular}{llcccc}
\toprule
Labeling Mode & System & MCC & Bal.\ Acc & F1 & $N$ \\
\midrule""")
    for sys in systems:
        df = load(sys, "ground_truth_results.csv")
        if df is None: continue
        rf = df[df["model"]=="RandomForest"]
        for _, row in rf.iterrows():
            m = row["label_mode"]
            lbl = "Default (Co-Change in Commit)" if m == "default" else "Strict-Module (Package Boundary)"
            print(f"{lbl} & {sys} & {float(row['mcc']):.4f} & {float(row['balanced_acc']):.4f} & "
                  f"{float(row['f1']):.4f} & {int(row['n'])} \\\\")
        print(r"\midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["tuxguitar","Ctags"])
    p.add_argument("--latex", action="store_true", help="Print LaTeX table code")
    args = p.parse_args()

    print_baselines_table(args.systems)
    print_ablation_table(args.systems)
    print_seed_table(args.systems)
    print_param_table(args.systems)
    print_gt_table(args.systems)

    if args.latex:
        print_latex_tables(args.systems)


if __name__ == "__main__":
    main()

