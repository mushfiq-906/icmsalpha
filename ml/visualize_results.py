"""
visualize_results.py - Generate all analysis graphs for thesis.

Usage:
  python ml/visualize_results.py
  python ml/visualize_results.py --datasets Ctags_Type3 Jmol_Type3
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import OrderedDict

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
COLORS = ["#2563EB","#DC2626","#059669","#D97706","#7C3AED"]
MODEL_ORDER = ["LightGBM","XGBoost","CatBoost","RandomForest","Ensemble"]

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#FAFAFA",
    "axes.grid":True,"grid.alpha":0.3,"font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,
})

def get_color(name):
    if name in MODEL_ORDER:
        return COLORS[MODEL_ORDER.index(name) % len(COLORS)]
    return COLORS[0]

# ── AUTO-DISCOVER DATASETS ──────────────────────────────────────────────────
def discover_datasets():
    """Find all dataset prefixes from model_comparison and walk_forward CSVs."""
    prefixes = set()
    for f in RESULTS_DIR.glob("*_model_comparison.csv"):
        prefixes.add(f.stem.replace("_model_comparison",""))
    for f in RESULTS_DIR.glob("walk_forward_metrics_*.csv"):
        prefixes.add(f.stem.replace("walk_forward_metrics_",""))
    # Also check non-prefixed files
    if (RESULTS_DIR/"model_comparison.csv").exists():
        prefixes.add("default")
    return sorted(prefixes)

def load_static(prefix):
    if prefix == "default":
        p = RESULTS_DIR / "model_comparison.csv"
    else:
        p = RESULTS_DIR / f"{prefix}_model_comparison.csv"
    if not p.exists(): return None
    return pd.read_csv(p)

def load_wf_metrics(prefix):
    if prefix == "default":
        return None
    p = RESULTS_DIR / f"walk_forward_metrics_{prefix}.csv"
    if not p.exists(): return None
    return pd.read_csv(p)

def load_wf_preds(prefix):
    if prefix == "default":
        return None
    p = RESULTS_DIR / f"walk_forward_predictions_{prefix}.csv"
    if not p.exists(): return None
    return pd.read_csv(p)

def load_forecast_csv(prefix):
    """Try to find the forecast dataset CSV for a given prefix."""
    base = Path(__file__).resolve().parent.parent / "WorkFolder"
    parts = prefix.split("_")
    if len(parts) >= 2:
        system, ctype = parts[0], parts[1]
        p = base/system/"Datasets"/"CloneGenealogy"/f"{ctype}_Block_forecast_dataset.csv"
        if p.exists(): return pd.read_csv(p)
    return None

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: DATASET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_dataset_analysis(prefix, outdir):
    df = load_forecast_csv(prefix)
    if df is None:
        print(f"  [SKIP] No forecast CSV for {prefix}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Dataset Analysis — {prefix}", fontsize=15, fontweight="bold")

    # 1a: Label distribution pie
    ax = axes[0,0]
    vc = df["will_diverge"].value_counts()
    labels = ["Co-change (0)","Independent (1)"]
    cols = ["#2563EB","#DC2626"]
    ax.pie(vc.values, labels=labels, colors=cols, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize":10})
    ax.set_title("Label Distribution")

    # 1b: Feature correlation heatmap (top 15 features by variance)
    ax = axes[0,1]
    excl = {"Revision","gcid1","gcid2","classid","will_diverge","will_diverge_decay"}
    feat_cols = [c for c in df.columns if c not in excl and df[c].dtype in [np.float64,np.int64,np.float32]]
    top = df[feat_cols].var().nlargest(12).index.tolist()
    if len(top) > 2:
        corr = df[top].corr()
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(top))); ax.set_yticks(range(len(top)))
        ax.set_xticklabels([t[:12] for t in top], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([t[:12] for t in top], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Feature Correlation (top 12 by variance)")

    # 1c: Revision distribution histogram
    ax = axes[1,0]
    ax.hist(df["Revision"], bins=40, color="#2563EB", alpha=0.7, edgecolor="white")
    cut = int(len(df)*0.7)
    cut_rev = df.sort_values("Revision").iloc[cut]["Revision"]
    ax.axvline(cut_rev, color="red", ls="--", lw=1.5, label=f"70/30 split (rev {int(cut_rev)})")
    ax.set_xlabel("Revision"); ax.set_ylabel("Count")
    ax.set_title("Event Distribution Over Revisions"); ax.legend(fontsize=9)

    # 1d: Label drift over revision buckets
    ax = axes[1,1]
    df2 = df.copy()
    df2["bucket"] = pd.qcut(df2["Revision"], q=10, duplicates="drop")
    drift = df2.groupby("bucket", observed=True)["will_diverge"].mean()
    drift.plot(kind="bar", ax=ax, color="#059669", edgecolor="white")
    ax.set_title("Label Drift (positive rate by revision bucket)")
    ax.set_ylabel("Fraction independent"); ax.set_xlabel("Revision bucket")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(outdir/f"{prefix}_dataset_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(); print(f"  -> {prefix}_dataset_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: PER-MODEL RESULTS (static split)
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_model_static(prefix, outdir):
    df = load_static(prefix)
    if df is None:
        print(f"  [SKIP] No static results for {prefix}")
        return

    models = df["Model"].tolist()
    metrics = ["MCC","AUC-ROC","PR-AUC","F1","Precision","Recall"]
    avail = [m for m in metrics if m in df.columns]

    for model in models:
        row = df[df["Model"]==model].iloc[0]
        vals = [row[m] for m in avail]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(avail, vals, color=get_color(model), edgecolor="white", height=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.set_title(f"{prefix} — {model} (Static 70/30 Split)", fontweight="bold")
        ax.set_xlabel("Score")
        plt.tight_layout()
        fig.savefig(outdir/f"{prefix}_{model}_static_metrics.png", dpi=150)
        plt.close()
    print(f"  -> {prefix}_*_static_metrics.png ({len(models)} models)")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: PER-MODEL RESULTS (walk-forward)
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_model_wf(prefix, outdir):
    df = load_wf_metrics(prefix)
    if df is None:
        print(f"  [SKIP] No walk-forward metrics for {prefix}")
        return

    models = df["Model"].tolist()
    metrics = ["MCC","AUC-ROC","PR-AUC","Balanced_Acc","G_mean","F1"]
    avail = [m for m in metrics if m in df.columns]

    for model in models:
        row = df[df["Model"]==model].iloc[0]
        vals = [row[m] for m in avail]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(avail, vals, color=get_color(model), edgecolor="white", height=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.set_title(f"{prefix} — {model} (Walk-Forward)", fontweight="bold")
        ax.set_xlabel("Score")
        plt.tight_layout()
        fig.savefig(outdir/f"{prefix}_{model}_wf_metrics.png", dpi=150)
        plt.close()
    print(f"  -> {prefix}_*_wf_metrics.png ({len(models)} models)")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: ROLLING MCC OVER TIME (per model)
# ═══════════════════════════════════════════════════════════════════════════

def plot_rolling_mcc_per_model(prefix, outdir, window=50):
    preds = load_wf_preds(prefix)
    if preds is None:
        print(f"  [SKIP] No walk-forward predictions for {prefix}")
        return
    from sklearn.metrics import matthews_corrcoef

    for model in preds["Model"].unique():
        sub = preds[preds["Model"]==model].sort_values("Revision").reset_index(drop=True)
        n = len(sub)
        mccs = np.full(n, np.nan)
        for i in range(window, n+1):
            yw = sub["y_true"].values[i-window:i]
            pw = sub["pred"].values[i-window:i]
            if len(np.unique(yw)) > 1:
                mccs[i-1] = matthews_corrcoef(yw, pw)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(sub["Revision"].values, mccs, color=get_color(model), lw=1.2, alpha=0.85)
        ax.axhline(0, color="grey", ls="--", alpha=0.5)
        ax.set_title(f"{prefix} — {model}: Rolling MCC (window={window})", fontweight="bold")
        ax.set_xlabel("Revision"); ax.set_ylabel("MCC")
        plt.tight_layout()
        fig.savefig(outdir/f"{prefix}_{model}_rolling_mcc.png", dpi=150)
        plt.close()
    print(f"  -> {prefix}_*_rolling_mcc.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: CONFUSION MATRIX HEATMAPS (per model)
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(prefix, outdir):
    df = load_wf_metrics(prefix)
    if df is None: return
    needed = {"TP","FP","TN","FN"}
    if not needed.issubset(df.columns): return

    for _, row in df.iterrows():
        model = row["Model"]
        cm = np.array([[row["TN"],row["FP"]],[row["FN"],row["TP"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i,j])}", ha="center", va="center",
                        fontsize=14, color="white" if cm[i,j]>cm.max()*0.5 else "black")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Co-change","Independent"])
        ax.set_yticklabels(["Co-change","Independent"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{prefix} — {model}: Confusion Matrix", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        fig.savefig(outdir/f"{prefix}_{model}_confusion.png", dpi=150)
        plt.close()
    print(f"  -> {prefix}_*_confusion.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: MODEL COMPARISON GRAPHS
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison_grouped_bar(prefix, outdir):
    """Grouped bar chart comparing all models on key metrics."""
    # Try walk-forward first, fallback to static
    df = load_wf_metrics(prefix)
    label = "Walk-Forward"
    if df is None:
        df = load_static(prefix)
        label = "Static 70/30"
    if df is None: return

    metrics = ["MCC","AUC-ROC","PR-AUC","F1"]
    avail = [m for m in metrics if m in df.columns]
    if not avail: return

    models = df["Model"].tolist()
    x = np.arange(len(avail))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        row = df[df["Model"]==model].iloc[0]
        vals = [row[m] for m in avail]
        bars = ax.bar(x + i*width, vals, width, label=model,
                      color=get_color(model), edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_xticks(x + width*(len(models)-1)/2)
    ax.set_xticklabels(avail, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score"); ax.legend(loc="lower right")
    ax.set_title(f"{prefix} — Model Comparison ({label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outdir/f"{prefix}_comparison_grouped_bar.png", dpi=150)
    plt.close()
    print(f"  -> {prefix}_comparison_grouped_bar.png")

def plot_comparison_radar(prefix, outdir):
    """Radar/spider chart comparing models."""
    df = load_wf_metrics(prefix)
    if df is None: df = load_static(prefix)
    if df is None: return

    metrics = ["MCC","AUC-ROC","PR-AUC","F1"]
    avail = [m for m in metrics if m in df.columns]
    if len(avail) < 3: return

    models = df["Model"].tolist()
    N = len(avail)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, model in enumerate(models):
        row = df[df["Model"]==model].iloc[0]
        vals = [row[m] for m in avail] + [row[avail[0]]]
        ax.plot(angles, vals, "o-", lw=2, label=model, color=get_color(model), alpha=0.8)
        ax.fill(angles, vals, alpha=0.1, color=get_color(model))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(avail, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title(f"{prefix} — Model Comparison Radar", fontsize=13, fontweight="bold", y=1.08)
    plt.tight_layout()
    fig.savefig(outdir/f"{prefix}_comparison_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {prefix}_comparison_radar.png")

def plot_comparison_rolling_mcc(prefix, outdir, window=50):
    """All models' rolling MCC on one chart."""
    preds = load_wf_preds(prefix)
    if preds is None: return
    from sklearn.metrics import matthews_corrcoef

    fig, ax = plt.subplots(figsize=(13, 5))
    for model in preds["Model"].unique():
        sub = preds[preds["Model"]==model].sort_values("Revision").reset_index(drop=True)
        n = len(sub)
        mccs = np.full(n, np.nan)
        for i in range(window, n+1):
            yw = sub["y_true"].values[i-window:i]
            pw = sub["pred"].values[i-window:i]
            if len(np.unique(yw)) > 1:
                mccs[i-1] = matthews_corrcoef(yw, pw)
        ax.plot(sub["Revision"].values, mccs, lw=1.3, alpha=0.85,
                label=model, color=get_color(model))

    ax.axhline(0, color="grey", ls="--", alpha=0.5)
    ax.set_title(f"{prefix} — All Models: Rolling MCC (window={window})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Revision"); ax.set_ylabel("MCC")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(outdir/f"{prefix}_comparison_rolling_mcc.png", dpi=150)
    plt.close()
    print(f"  -> {prefix}_comparison_rolling_mcc.png")

def plot_comparison_recall_balance(prefix, outdir):
    """Compare recall for both classes across models."""
    df = load_wf_metrics(prefix)
    if df is None: return
    if "Recall_pos" not in df.columns: return

    models = df["Model"].tolist()
    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, df["Recall_neg"].values, w, label="Recall (co-change)", color="#2563EB", edgecolor="white")
    ax.bar(x + w/2, df["Recall_pos"].values, w, label="Recall (independent)", color="#DC2626", edgecolor="white")
    for i, (rn, rp) in enumerate(zip(df["Recall_neg"], df["Recall_pos"])):
        ax.text(i-w/2, rn+0.01, f"{rn:.3f}", ha="center", fontsize=9)
        ax.text(i+w/2, rp+0.01, f"{rp:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Recall"); ax.legend()
    ax.set_title(f"{prefix} — Per-Class Recall Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outdir/f"{prefix}_comparison_recall.png", dpi=150)
    plt.close()
    print(f"  -> {prefix}_comparison_recall.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: CROSS-DATASET COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_cross_dataset(prefixes, outdir):
    """Compare same model across different datasets."""
    if len(prefixes) < 2:
        print("  [SKIP] Need >= 2 datasets for cross-dataset comparison")
        return

    # Collect MCC per model per dataset
    data = {}
    for pfx in prefixes:
        df = load_wf_metrics(pfx)
        if df is None: df = load_static(pfx)
        if df is None: continue
        for _, row in df.iterrows():
            model = row["Model"]
            if model not in data: data[model] = {}
            data[model][pfx] = row.get("MCC", 0)

    if not data: return

    models = [m for m in MODEL_ORDER if m in data] + [m for m in data if m not in MODEL_ORDER]
    datasets = prefixes
    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        vals = [data[model].get(d, 0) for d in datasets]
        ax.bar(x + i*width, vals, width, label=model, color=get_color(model), edgecolor="white")

    ax.set_xticks(x + width*(len(models)-1)/2)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("MCC"); ax.legend(loc="best")
    ax.set_title("Cross-Dataset MCC Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outdir/"cross_dataset_mcc.png", dpi=150)
    plt.close()
    print(f"  -> cross_dataset_mcc.png")

def plot_cross_dataset_heatmap(prefixes, outdir):
    """Heatmap of MCC: rows=models, cols=datasets."""
    if len(prefixes) < 2: return

    rows, cols = [], prefixes
    mcc_data = []
    all_models = set()
    dfs = {}
    for pfx in prefixes:
        df = load_wf_metrics(pfx)
        if df is None: df = load_static(pfx)
        if df is not None:
            dfs[pfx] = df
            all_models.update(df["Model"].tolist())

    models = [m for m in MODEL_ORDER if m in all_models] + [m for m in all_models if m not in MODEL_ORDER]
    matrix = np.zeros((len(models), len(prefixes)))
    for j, pfx in enumerate(prefixes):
        if pfx not in dfs: continue
        df = dfs[pfx]
        for i, model in enumerate(models):
            r = df[df["Model"]==model]
            if len(r): matrix[i,j] = r.iloc[0].get("MCC", 0)

    if matrix.size == 0: return

    fig, ax = plt.subplots(figsize=(max(8,len(prefixes)*3), max(4,len(models)*0.8)))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    for i in range(len(models)):
        for j in range(len(prefixes)):
            ax.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center", fontsize=11,
                    color="white" if matrix[i,j]>matrix.max()*0.6 else "black")
    ax.set_xticks(range(len(prefixes))); ax.set_yticks(range(len(models)))
    ax.set_xticklabels(prefixes); ax.set_yticklabels(models)
    ax.set_title("MCC Heatmap: Models × Datasets", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(outdir/"cross_dataset_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> cross_dataset_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate all thesis visualization graphs")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset prefixes (e.g. Ctags_Type3 Jmol_Type3). Auto-discovers if omitted.")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    prefixes = args.datasets or discover_datasets()

    # Also check RESULT_*.csv files
    for f in RESULTS_DIR.glob("RESULT_*.csv"):
        tag = f.stem.replace("RESULT_","")
        if tag not in prefixes:
            prefixes.append(tag)

    print(f"Discovered datasets: {prefixes}")
    print(f"Output directory   : {PLOTS_DIR}\n")

    for pfx in prefixes:
        print(f"\n{'='*60}")
        print(f"  Processing: {pfx}")
        print(f"{'='*60}")

        print("\n[1] Dataset Analysis")
        plot_dataset_analysis(pfx, PLOTS_DIR)

        print("\n[2] Per-Model Static Results")
        plot_per_model_static(pfx, PLOTS_DIR)

        print("\n[3] Per-Model Walk-Forward Results")
        plot_per_model_wf(pfx, PLOTS_DIR)

        print("\n[4] Per-Model Rolling MCC")
        plot_rolling_mcc_per_model(pfx, PLOTS_DIR)

        print("\n[5] Confusion Matrices")
        plot_confusion_matrices(pfx, PLOTS_DIR)

        print("\n[6] Model Comparison — Grouped Bar")
        plot_comparison_grouped_bar(pfx, PLOTS_DIR)

        print("\n[7] Model Comparison — Radar")
        plot_comparison_radar(pfx, PLOTS_DIR)

        print("\n[8] Model Comparison — Rolling MCC (all models)")
        plot_comparison_rolling_mcc(pfx, PLOTS_DIR)

        print("\n[9] Model Comparison — Recall Balance")
        plot_comparison_recall_balance(pfx, PLOTS_DIR)

    # Cross-dataset comparisons
    if len(prefixes) >= 2:
        print(f"\n{'='*60}")
        print(f"  Cross-Dataset Comparisons")
        print(f"{'='*60}")
        plot_cross_dataset(prefixes, PLOTS_DIR)
        plot_cross_dataset_heatmap(prefixes, PLOTS_DIR)

    print(f"\n[DONE] All plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
