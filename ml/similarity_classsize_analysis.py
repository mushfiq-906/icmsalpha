#!/usr/bin/env python3
"""
Clone Similarity & Class Size as Dependency Triggers
=====================================================
Standalone interpretability analysis that investigates whether
syntactic similarity and clone class size drive dependent vs.
independent clone fragment change behaviour.

Currently implemented for: Ctags (Type3_Block)

Data sources:
  - rev_pair.csv (per-pair events with similarity, classSize)

Outputs (in ml/results/similarity_classsize_analysis/{SYSTEM}/):
  - {SYSTEM}_similarity_classsize.csv
  - {SYSTEM}_simclass_stats.txt
  - {SYSTEM}_similarity_bucket_dependency.png
  - {SYSTEM}_classsize_bucket_dependency.png
  - {SYSTEM}_similarity_scatter.png
  - {SYSTEM}_simclass_interaction_heatmap.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
SYSTEM = "tuxguitar"
CLONE_TYPE = "Type3_Block"
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "WorkFolder", SYSTEM, "Datasets", "CloneGenealogy"
)
REV_PAIR = os.path.join(BASE_DIR, f"{CLONE_TYPE}_rev_pair.csv")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "similarity_classsize_analysis", SYSTEM
)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
})
PALETTE = sns.color_palette("mako", 6)
DEP_COLOR = "#2ecc71"
INDEP_COLOR = "#e74c3c"


# ──────────────────────────────────────────────────────────────────────
# 1. Load & label data
# ──────────────────────────────────────────────────────────────────────
def load_data():
    """Load rev_pair.csv and derive dependency labels.

    Returns:
        pd.DataFrame: pair-revision events with dep_label, similarity buckets,
                       and classSize buckets.
    """
    df = pd.read_csv(REV_PAIR)
    df["dep_label"] = (
        (df["changeType1"] != "U") & (df["changeType2"] != "U")
    ).astype(int)

    # Similarity buckets
    df["sim_bucket"] = pd.cut(
        df["similarity"], bins=[0, 70, 80, 90, 95, 101],
        labels=["<=70", "71-80", "81-90", "91-95", "96-100"], right=True,
    )

    # Class size buckets
    df["class_bucket"] = pd.cut(
        df["classSize"], bins=[0, 2, 5, 10, 20, 1000],
        labels=["2 (pair)", "3-5", "6-10", "11-20", "21+"], right=True,
    )
    return df


# ──────────────────────────────────────────────────────────────────────
# 2. Aggregate statistics
# ──────────────────────────────────────────────────────────────────────
def compute_stats(df):
    """Compute dependency rates by similarity and class size buckets.

    Args:
        df: DataFrame with dep_label, sim_bucket, class_bucket columns.

    Returns:
        dict: Keys 'by_similarity', 'by_classsize', 'cross' with aggregated DataFrames.
    """
    results = {}

    # By similarity bucket
    sim = df.groupby("sim_bucket", observed=False).agg(
        total=("dep_label", "count"), dependent=("dep_label", "sum"),
    ).reset_index()
    sim["independent"] = sim["total"] - sim["dependent"]
    sim["dep_rate"] = np.where(sim["total"] > 0, sim["dependent"] / sim["total"], 0)
    results["by_similarity"] = sim

    # By class size bucket
    cls = df.groupby("class_bucket", observed=False).agg(
        total=("dep_label", "count"), dependent=("dep_label", "sum"),
    ).reset_index()
    cls["independent"] = cls["total"] - cls["dependent"]
    cls["dep_rate"] = np.where(cls["total"] > 0, cls["dependent"] / cls["total"], 0)
    results["by_classsize"] = cls

    # Cross: similarity x classSize
    cross = df.groupby(["sim_bucket", "class_bucket"], observed=False).agg(
        total=("dep_label", "count"), dependent=("dep_label", "sum"),
    ).reset_index()
    cross["dep_rate"] = np.where(cross["total"] > 0, cross["dependent"] / cross["total"], 0)
    results["cross"] = cross

    return results


# ──────────────────────────────────────────────────────────────────────
# 3. Statistical tests
# ──────────────────────────────────────────────────────────────────────
def run_statistical_tests(df):
    """Run correlation and regression tests for similarity and classSize.

    Args:
        df: DataFrame with dep_label, similarity, classSize columns.

    Returns:
        dict: Test results including correlations, Mann-Whitney, logistic regression.
    """
    results = {}

    # Point-biserial: similarity <-> dep_label
    r_sim, p_sim = stats.pointbiserialr(df["dep_label"], df["similarity"])
    results["pb_similarity_r"] = r_sim
    results["pb_similarity_p"] = p_sim

    # Point-biserial: classSize <-> dep_label
    r_cls, p_cls = stats.pointbiserialr(df["dep_label"], df["classSize"])
    results["pb_classSize_r"] = r_cls
    results["pb_classSize_p"] = p_cls

    # Spearman rank correlation
    rho_sim, p_rho_sim = stats.spearmanr(df["similarity"], df["dep_label"])
    results["spearman_similarity_rho"] = rho_sim
    results["spearman_similarity_p"] = p_rho_sim

    rho_cls, p_rho_cls = stats.spearmanr(df["classSize"], df["dep_label"])
    results["spearman_classSize_rho"] = rho_cls
    results["spearman_classSize_p"] = p_rho_cls

    # Mann-Whitney U: similarity for dep vs indep
    dep_sim = df.loc[df["dep_label"] == 1, "similarity"]
    ind_sim = df.loc[df["dep_label"] == 0, "similarity"]
    if len(dep_sim) > 0 and len(ind_sim) > 0:
        u, p_mw = stats.mannwhitneyu(dep_sim, ind_sim, alternative="two-sided")
        results["mw_similarity_U"] = u
        results["mw_similarity_p"] = p_mw
        results["dep_similarity_mean"] = float(dep_sim.mean())
        results["dep_similarity_median"] = float(dep_sim.median())
        results["ind_similarity_mean"] = float(ind_sim.mean())
        results["ind_similarity_median"] = float(ind_sim.median())

    # Mann-Whitney U: classSize for dep vs indep
    dep_cls = df.loc[df["dep_label"] == 1, "classSize"]
    ind_cls = df.loc[df["dep_label"] == 0, "classSize"]
    if len(dep_cls) > 0 and len(ind_cls) > 0:
        u2, p_mw2 = stats.mannwhitneyu(dep_cls, ind_cls, alternative="two-sided")
        results["mw_classSize_U"] = u2
        results["mw_classSize_p"] = p_mw2
        results["dep_classSize_mean"] = float(dep_cls.mean())
        results["dep_classSize_median"] = float(dep_cls.median())
        results["ind_classSize_mean"] = float(ind_cls.mean())
        results["ind_classSize_median"] = float(ind_cls.median())

    # Threshold analysis: dep rate for sim >= threshold
    results["threshold_analysis"] = {}
    for t in [70, 75, 80, 85, 90, 95]:
        subset = df[df["similarity"] >= t]
        if len(subset) > 0:
            dr = subset["dep_label"].mean()
            results["threshold_analysis"][t] = {"n": len(subset), "dep_rate": dr}

    # Small vs large class comparison
    small = df[df["classSize"] == 2]
    large = df[df["classSize"] >= 5]
    if len(small) > 0 and len(large) > 0:
        results["small_class_dep_rate"] = float(small["dep_label"].mean())
        results["large_class_dep_rate"] = float(large["dep_label"].mean())
        results["small_class_n"] = len(small)
        results["large_class_n"] = len(large)

    # Logistic regression
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        features = df[["similarity", "classSize"]].copy().dropna()
        y = df.loc[features.index, "dep_label"]
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(features)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_sc, y)
        results["logistic_accuracy"] = lr.score(X_sc, y)
        results["logistic_coefs"] = dict(zip(features.columns, lr.coef_[0]))
        # Unscaled for OR
        lr2 = LogisticRegression(max_iter=1000, random_state=42)
        lr2.fit(features, y)
        results["odds_ratios"] = {k: np.exp(v) for k, v in zip(features.columns, lr2.coef_[0])}
    except ImportError:
        pass

    return results


# ──────────────────────────────────────────────────────────────────────
# 4. Visualisations
# ──────────────────────────────────────────────────────────────────────
def plot_similarity_bars(struct_stats, out_dir):
    """Bar chart: dependency rate by similarity bucket.

    Args:
        struct_stats: dict from compute_stats().
        out_dir: output directory path.
    """
    sim = struct_stats["by_similarity"]
    sim = sim[sim["total"] > 0].copy().reset_index(drop=True)
    if sim.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sim))
    bars = ax.bar(x, sim["dep_rate"], 0.5, color=PALETTE[2], edgecolor="white")
    for i, (_, row) in enumerate(sim.iterrows()):
        ax.text(i, row["dep_rate"] + 0.01,
                f'{row["dep_rate"]:.1%}\n(n={row["total"]:,})',
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(sim["sim_bucket"])
    ax.set_xlabel("Similarity Score Range")
    ax.set_ylabel("Dependency Rate")
    ax.set_title(f"Dependency Rate by Clone Similarity - {SYSTEM} ({CLONE_TYPE})")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(sim["dep_rate"].max() + 0.15, 1.05))
    ax.grid(axis="y", alpha=0.2)
    sns.despine()
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_similarity_bucket_dependency.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_similarity_bucket_dependency.png")


def plot_classsize_bars(struct_stats, out_dir):
    """Bar chart: dependency rate by class size bucket.

    Args:
        struct_stats: dict from compute_stats().
        out_dir: output directory path.
    """
    cls = struct_stats["by_classsize"]
    cls = cls[cls["total"] > 0].copy().reset_index(drop=True)
    if cls.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cls))
    bars = ax.bar(x, cls["dep_rate"], 0.5, color=PALETTE[3], edgecolor="white")
    for i, (_, row) in enumerate(cls.iterrows()):
        ax.text(i, row["dep_rate"] + 0.01,
                f'{row["dep_rate"]:.1%}\n(n={row["total"]:,})',
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cls["class_bucket"])
    ax.set_xlabel("Clone Class Size")
    ax.set_ylabel("Dependency Rate")
    ax.set_title(f"Dependency Rate by Clone Class Size - {SYSTEM} ({CLONE_TYPE})")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(cls["dep_rate"].max() + 0.15, 1.05))
    ax.grid(axis="y", alpha=0.2)
    sns.despine()
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_classsize_bucket_dependency.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_classsize_bucket_dependency.png")


def plot_similarity_scatter(df, test_results, out_dir):
    """Scatter/strip plot: similarity vs dep_label with jitter.

    Args:
        df: DataFrame with similarity and dep_label.
        test_results: dict from run_statistical_tests().
        out_dir: output directory path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: similarity distribution for dep vs indep
    ax = axes[0]
    dep_vals = df.loc[df["dep_label"] == 1, "similarity"]
    ind_vals = df.loc[df["dep_label"] == 0, "similarity"]
    ax.hist(ind_vals, bins=30, alpha=0.6, label=f"Independent (n={len(ind_vals):,})",
            color=INDEP_COLOR, edgecolor="white")
    ax.hist(dep_vals, bins=30, alpha=0.6, label=f"Dependent (n={len(dep_vals):,})",
            color=DEP_COLOR, edgecolor="white")
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Count")
    r = test_results.get("pb_similarity_r", 0)
    p = test_results.get("pb_similarity_p", 1)
    ax.set_title(f"Similarity Distribution\nPoint-biserial r={r:.4f}, p={p:.2e}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    # Right: classSize distribution for dep vs indep
    ax = axes[1]
    dep_cls = df.loc[df["dep_label"] == 1, "classSize"]
    ind_cls = df.loc[df["dep_label"] == 0, "classSize"]
    ax.hist(ind_cls, bins=30, alpha=0.6, label=f"Independent (n={len(ind_cls):,})",
            color=INDEP_COLOR, edgecolor="white")
    ax.hist(dep_cls, bins=30, alpha=0.6, label=f"Dependent (n={len(dep_cls):,})",
            color=DEP_COLOR, edgecolor="white")
    ax.set_xlabel("Class Size")
    ax.set_ylabel("Count")
    r2 = test_results.get("pb_classSize_r", 0)
    p2 = test_results.get("pb_classSize_p", 1)
    ax.set_title(f"Class Size Distribution\nPoint-biserial r={r2:.4f}, p={p2:.2e}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Similarity & Class Size Distributions - {SYSTEM}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_similarity_scatter.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_similarity_scatter.png")


def plot_interaction_heatmap(struct_stats, out_dir):
    """Heatmap: dep rate by similarity bucket x class size bucket.

    Args:
        struct_stats: dict from compute_stats().
        out_dir: output directory path.
    """
    cross = struct_stats["cross"]
    cross = cross[cross["total"] > 0].copy()
    if cross.empty:
        return
    pivot = cross.pivot_table(
        index="class_bucket", columns="sim_bucket",
        values="dep_rate", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax,
                linewidths=0.5, vmin=0)
    ax.set_xlabel("Similarity Bucket")
    ax.set_ylabel("Class Size Bucket")
    ax.set_title(f"Interaction: Similarity x Class Size -> Dep Rate - {SYSTEM}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_simclass_interaction_heatmap.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_simclass_interaction_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  Similarity & Class Size Analysis - {SYSTEM} ({CLONE_TYPE})")
    print("=" * 60 + "\n")

    print("[1/5] Loading data...")
    df = load_data()
    print(f"  Events: {len(df):,}")
    print(f"  Dep events: {df['dep_label'].sum():,} / {len(df):,} ({df['dep_label'].mean():.1%})")
    print(f"  Similarity: min={df['similarity'].min()}, max={df['similarity'].max()}, "
          f"mean={df['similarity'].mean():.1f}")
    print(f"  ClassSize: min={df['classSize'].min()}, max={df['classSize'].max()}, "
          f"mean={df['classSize'].mean():.1f}")

    print("\n[2/5] Computing aggregate statistics...")
    struct_stats = compute_stats(df)
    print("\n  Dep rate by similarity:")
    print(struct_stats["by_similarity"].to_string(index=False))
    print("\n  Dep rate by class size:")
    print(struct_stats["by_classsize"].to_string(index=False))

    print("\n[3/5] Saving CSV...")
    export_cols = ["revision", "gcid1", "gcid2", "classId", "similarity", "classSize",
                   "sim_bucket", "class_bucket", "sameFile", "depth", "dep_label"]
    available = [c for c in export_cols if c in df.columns]
    csv_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_similarity_classsize.csv")
    df[available].to_csv(csv_path, index=False)
    print(f"  [OK] Saved {csv_path}")

    print("\n[4/5] Running statistical tests...")
    test_results = run_statistical_tests(df)

    lines = []
    lines.append(f"Similarity & Class Size Analysis - {SYSTEM} ({CLONE_TYPE})")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1. Point-biserial correlations with dep_label")
    lines.append(f"   similarity: r={test_results['pb_similarity_r']:.4f}, p={test_results['pb_similarity_p']:.2e}")
    lines.append(f"   classSize:  r={test_results['pb_classSize_r']:.4f}, p={test_results['pb_classSize_p']:.2e}")
    lines.append("")
    lines.append("2. Spearman rank correlations")
    lines.append(f"   similarity: rho={test_results['spearman_similarity_rho']:.4f}, p={test_results['spearman_similarity_p']:.2e}")
    lines.append(f"   classSize:  rho={test_results['spearman_classSize_rho']:.4f}, p={test_results['spearman_classSize_p']:.2e}")
    lines.append("")

    lines.append("3. Mann-Whitney U: similarity (dep vs indep)")
    if "mw_similarity_U" in test_results:
        lines.append(f"   U={test_results['mw_similarity_U']:.0f}, p={test_results['mw_similarity_p']:.2e}")
        lines.append(f"   Dep mean={test_results['dep_similarity_mean']:.2f}, median={test_results['dep_similarity_median']:.0f}")
        lines.append(f"   Ind mean={test_results['ind_similarity_mean']:.2f}, median={test_results['ind_similarity_median']:.0f}")
    lines.append("")

    lines.append("4. Mann-Whitney U: classSize (dep vs indep)")
    if "mw_classSize_U" in test_results:
        lines.append(f"   U={test_results['mw_classSize_U']:.0f}, p={test_results['mw_classSize_p']:.2e}")
        lines.append(f"   Dep mean={test_results['dep_classSize_mean']:.1f}, median={test_results['dep_classSize_median']:.0f}")
        lines.append(f"   Ind mean={test_results['ind_classSize_mean']:.1f}, median={test_results['ind_classSize_median']:.0f}")
    lines.append("")

    lines.append("5. Similarity threshold analysis")
    for t, info in test_results.get("threshold_analysis", {}).items():
        lines.append(f"   sim >= {t}: n={info['n']:,}, dep_rate={info['dep_rate']:.1%}")
    lines.append("")

    lines.append("6. Small vs Large class comparison")
    if "small_class_dep_rate" in test_results:
        lines.append(f"   classSize=2 (n={test_results['small_class_n']:,}): dep_rate={test_results['small_class_dep_rate']:.1%}")
        lines.append(f"   classSize>=5 (n={test_results['large_class_n']:,}): dep_rate={test_results['large_class_dep_rate']:.1%}")
    lines.append("")

    if "odds_ratios" in test_results:
        lines.append("7. Logistic Regression: dep ~ similarity + classSize")
        lines.append(f"   Accuracy: {test_results['logistic_accuracy']:.4f}")
        for feat, or_val in test_results["odds_ratios"].items():
            lines.append(f"   OR({feat}): {or_val:.4f}")

    stats_text = "\n".join(lines)
    print(stats_text)
    stats_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_simclass_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\n  [OK] Saved {stats_path}")

    print("\n[5/5] Generating plots...")
    plot_similarity_bars(struct_stats, RESULTS_DIR)
    plot_classsize_bars(struct_stats, RESULTS_DIR)
    plot_similarity_scatter(df, test_results, RESULTS_DIR)
    plot_interaction_heatmap(struct_stats, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("  DONE - Similarity & ClassSize analysis complete for", SYSTEM)
    print("=" * 60)


if __name__ == "__main__":
    main()
