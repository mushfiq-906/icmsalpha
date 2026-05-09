#!/usr/bin/env python3
"""
File/Folder Structural Proximity Analysis
==========================================
Standalone interpretability analysis that investigates whether
file/folder co-location of clone fragments drives dependent vs.
independent change behaviour.

Currently implemented for: Ctags (Type3_Block)

Data sources:
  - rev_pair.csv      (per-pair events with sameFile, depth columns)
  - rev_fragment.csv  (per-fragment events with filePath for enrichment)

Outputs (in ml/results/):
  - Ctags_file-folder-analysis.csv
  - Ctags_filefolder_distance.png
  - Ctags_filefolder_samefile.png
  - Ctags_filefolder_violin.png
  - Ctags_filefolder_heatmap.png
  - Ctags_filefolder_stats.txt
"""

import os
import sys
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
SYSTEM = "Ctags"
CLONE_TYPE = "Type3_Block"
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "WorkFolder", SYSTEM, "Datasets", "CloneGenealogy"
)
REV_PAIR = os.path.join(BASE_DIR, f"{CLONE_TYPE}_rev_pair.csv")
REV_FRAGMENT = os.path.join(BASE_DIR, f"{CLONE_TYPE}_rev_fragment.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "file_folder_analysis", SYSTEM)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})
PALETTE = sns.color_palette("mako", 6)
DEP_COLOR = "#2ecc71"
INDEP_COLOR = "#e74c3c"


# ──────────────────────────────────────────────────────────────────────
# 1. Load & enrich data
# ──────────────────────────────────────────────────────────────────────
def load_data():
    """Load rev_pair (has sameFile, depth) and enrich with file paths from rev_fragment."""
    rev_pair = pd.read_csv(REV_PAIR)
    rev_frag = pd.read_csv(REV_FRAGMENT)

    # Dep label: both sides changed
    rev_pair["dep_label"] = (
        (rev_pair["changeType1"] != "U") & (rev_pair["changeType2"] != "U")
    ).astype(int)

    # --- Enrich with file paths from rev_fragment ---
    path_map = rev_frag[["revision", "gcid", "filePath"]].drop_duplicates()

    merged = rev_pair.merge(
        path_map.rename(columns={"gcid": "gcid1", "filePath": "filepath1"}),
        on=["revision", "gcid1"],
        how="left",
    )
    merged = merged.merge(
        path_map.rename(columns={"gcid": "gcid2", "filePath": "filepath2"}),
        on=["revision", "gcid2"],
        how="left",
    )

    # Derive folder info from file paths
    merged["folder1"] = merged["filepath1"].apply(
        lambda x: os.path.dirname(str(x)).replace("\\", "/") if pd.notna(x) else ""
    )
    merged["folder2"] = merged["filepath2"].apply(
        lambda x: os.path.dirname(str(x)).replace("\\", "/") if pd.notna(x) else ""
    )
    merged["same_folder"] = (merged["folder1"] == merged["folder2"]).astype(int)

    # Use the 'depth' column from rev_pair as file distance proxy
    # depth=0 means same directory, higher = more levels apart
    merged["file_distance"] = merged["depth"]

    # Bucket file_distance
    merged["distance_bucket"] = pd.cut(
        merged["file_distance"].fillna(0).astype(int),
        bins=[-1, 0, 1, 2, 3, 100],
        labels=["0 (same dir)", "1", "2", "3", "4+"],
        right=True,
    )

    return merged


# ──────────────────────────────────────────────────────────────────────
# 2. Aggregate statistics
# ──────────────────────────────────────────────────────────────────────
def compute_structural_stats(df):
    """Compute dependency rate by structural features."""
    results = {}

    # --- By sameFile ---
    sf = df.groupby("sameFile").agg(
        total=("dep_label", "count"),
        dependent=("dep_label", "sum"),
    ).reset_index()
    sf["independent"] = sf["total"] - sf["dependent"]
    sf["dep_rate"] = sf["dependent"] / sf["total"]
    results["by_sameFile"] = sf

    # --- By file_distance bucket ---
    fd = df.groupby("distance_bucket", observed=False).agg(
        total=("dep_label", "count"),
        dependent=("dep_label", "sum"),
    ).reset_index()
    fd["independent"] = fd["total"] - fd["dependent"]
    fd["dep_rate"] = np.where(fd["total"] > 0, fd["dependent"] / fd["total"], 0)
    results["by_distance"] = fd

    # --- By same_folder ---
    sfolder = df.groupby("same_folder").agg(
        total=("dep_label", "count"),
        dependent=("dep_label", "sum"),
    ).reset_index()
    sfolder["independent"] = sfolder["total"] - sfolder["dependent"]
    sfolder["dep_rate"] = sfolder["dependent"] / sfolder["total"]
    results["by_same_folder"] = sfolder

    # --- sameFile x distance_bucket ---
    cross = df.groupby(["sameFile", "distance_bucket"], observed=False).agg(
        total=("dep_label", "count"),
        dependent=("dep_label", "sum"),
    ).reset_index()
    cross["dep_rate"] = np.where(
        cross["total"] > 0, cross["dependent"] / cross["total"], 0
    )
    results["cross_file_distance"] = cross

    return results


# ──────────────────────────────────────────────────────────────────────
# 3. Statistical tests
# ──────────────────────────────────────────────────────────────────────
def run_statistical_tests(df):
    """Run chi-squared, Mann-Whitney U, point-biserial, logistic regression."""
    results = {}

    # --- 3a. Chi-squared: sameFile x dep_label ---
    ct = pd.crosstab(df["sameFile"], df["dep_label"])
    if ct.shape == (2, 2):
        chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        cramers_v = np.sqrt(chi2 / n) if n > 0 else 0
        results["sameFile_chi2"] = chi2
        results["sameFile_chi2_pvalue"] = p_chi2
        results["sameFile_cramersV"] = cramers_v

        odds, p_fisher = stats.fisher_exact(ct.values)
        results["sameFile_fisher_OR"] = odds
        results["sameFile_fisher_pvalue"] = p_fisher
    results["sameFile_contingency"] = ct

    # --- 3b. Mann-Whitney U: file_distance for dep vs. indep ---
    valid = df.dropna(subset=["file_distance"])
    dep_dist = valid.loc[valid["dep_label"] == 1, "file_distance"]
    indep_dist = valid.loc[valid["dep_label"] == 0, "file_distance"]
    if len(dep_dist) > 0 and len(indep_dist) > 0:
        u_stat, p_mw = stats.mannwhitneyu(dep_dist, indep_dist, alternative="two-sided")
        results["mannwhitney_U"] = u_stat
        results["mannwhitney_pvalue"] = p_mw
        results["dep_distance_median"] = dep_dist.median()
        results["dep_distance_mean"] = dep_dist.mean()
        results["indep_distance_median"] = indep_dist.median()
        results["indep_distance_mean"] = indep_dist.mean()

    # --- 3c. Point-biserial correlation: file_distance <-> dep_label ---
    valid2 = df.dropna(subset=["file_distance"])
    if len(valid2) > 2:
        r_pb, p_pb = stats.pointbiserialr(valid2["dep_label"], valid2["file_distance"])
        results["pointbiserial_r"] = r_pb
        results["pointbiserial_pvalue"] = p_pb

    # --- 3d. Logistic regression: dep_label ~ sameFile + file_distance ---
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        features = df[["sameFile"]].copy()
        features["file_distance"] = df["file_distance"].fillna(0)
        features = features.dropna()
        y = df.loc[features.index, "dep_label"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)

        coef_dict = dict(zip(features.columns, lr.coef_[0]))
        results["logistic_coefficients"] = coef_dict
        results["logistic_intercept"] = lr.intercept_[0]
        results["logistic_accuracy"] = lr.score(X_scaled, y)

        # Odds ratios (using unscaled re-fit for interpretability)
        lr2 = LogisticRegression(max_iter=1000, random_state=42)
        lr2.fit(features, y)
        or_dict = {k: np.exp(v) for k, v in zip(features.columns, lr2.coef_[0])}
        results["odds_ratios"] = or_dict
    except ImportError:
        pass

    return results


# ──────────────────────────────────────────────────────────────────────
# 4. Visualisations
# ──────────────────────────────────────────────────────────────────────
def plot_distance_bars(struct_stats, out_dir):
    """Bar chart: dependency rate by file_distance bucket."""
    fd = struct_stats["by_distance"]
    fd = fd[fd["total"] > 0].copy().reset_index(drop=True)
    if fd.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(fd))
    w = 0.5

    bars = ax.bar(x, fd["dep_rate"], w, color=PALETTE[2], edgecolor="white",
                  linewidth=0.5)

    # Annotate counts
    for i, (_, row) in enumerate(fd.iterrows()):
        ax.text(i, row["dep_rate"] + 0.01,
                f'{row["dep_rate"]:.1%}\n(n={row["total"]:,})',
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(fd["distance_bucket"])
    ax.set_xlabel("File Distance (depth: directory levels apart)")
    ax.set_ylabel("Dependency Rate")
    ax.set_title(f"Dependency Rate by File Distance - {SYSTEM} ({CLONE_TYPE})")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(fd["dep_rate"].max() + 0.15, 1.05))
    ax.grid(axis="y", alpha=0.2)
    sns.despine()

    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_filefolder_distance.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_filefolder_distance.png")


def plot_samefile_bars(struct_stats, test_results, out_dir):
    """Side-by-side bars: same file vs. cross-file dep rate."""
    sf = struct_stats["by_sameFile"]

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Cross-file", "Same file"]
    colors = [INDEP_COLOR, DEP_COLOR]

    for i, (_, row) in enumerate(sf.iterrows()):
        ax.bar(i, row["dep_rate"], 0.5, color=colors[i], edgecolor="white",
               linewidth=0.5)
        ax.text(i, row["dep_rate"] + 0.01,
                f'{row["dep_rate"]:.1%}\n(n={row["total"]:,})',
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    p_val = test_results.get("sameFile_fisher_pvalue",
                              test_results.get("sameFile_chi2_pvalue"))
    or_val = test_results.get("sameFile_fisher_OR")

    title = f"Same-File vs. Cross-File Dependency - {SYSTEM}"
    if p_val is not None and or_val is not None:
        title += f"\nFisher OR={or_val:.3f}  |  p={p_val:.2e}"

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Dependency Rate")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(sf["dep_rate"].max() + 0.15, 1.05))
    ax.grid(axis="y", alpha=0.2)
    sns.despine()

    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_filefolder_samefile.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_filefolder_samefile.png")


def plot_distance_violin(df, test_results, out_dir):
    """Violin plot: file_distance distribution for dep vs. indep."""
    valid = df.dropna(subset=["file_distance"]).copy()
    if valid.empty:
        return

    dep_vals = valid.loc[valid["dep_label"] == 1, "file_distance"].values
    indep_vals = valid.loc[valid["dep_label"] == 0, "file_distance"].values

    if len(dep_vals) == 0 or len(indep_vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    parts = ax.violinplot(
        [dep_vals, indep_vals],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
    )

    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([DEP_COLOR, INDEP_COLOR][i])
        pc.set_alpha(0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Dependent", "Independent"])
    ax.set_ylabel("File Distance (depth)")

    p_mw = test_results.get("mannwhitney_pvalue")
    med_dep = test_results.get("dep_distance_median")
    med_ind = test_results.get("indep_distance_median")

    title = f"File Distance Distribution - {SYSTEM}\n"
    if p_mw is not None:
        title += f"Mann-Whitney p={p_mw:.2e} | Median: dep={med_dep}, indep={med_ind}"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    sns.despine()

    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_filefolder_violin.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_filefolder_violin.png")


def plot_cross_heatmap(struct_stats, out_dir):
    """Heatmap: dep rate by sameFile x distance bucket."""
    cross = struct_stats["cross_file_distance"]
    cross = cross[cross["total"] > 0].copy()

    if cross.empty:
        return

    pivot = cross.pivot_table(
        index="sameFile", columns="distance_bucket",
        values="dep_rate", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax,
        yticklabels=["Cross-file", "Same file"],
        linewidths=0.5,
    )
    ax.set_xlabel("File Distance Bucket (depth)")
    ax.set_ylabel("Same File")
    ax.set_title(f"Dependency Rate: Same-File x File Distance - {SYSTEM}")

    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_filefolder_heatmap.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_filefolder_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  File/Folder Structural Proximity - {SYSTEM} ({CLONE_TYPE})")
    print("=" * 60 + "\n")

    # Load
    print("[1/5] Loading and enriching data...")
    df = load_data()
    print(f"  Events:       {len(df):,}")
    print(f"  Dep events:   {df['dep_label'].sum():,} / {len(df):,} "
          f"({df['dep_label'].mean():.1%})")
    print(f"  Same-file:    {df['sameFile'].sum():,} / {len(df):,}")
    n_matched = df["filepath1"].notna().sum()
    print(f"  Path enriched: {n_matched:,} / {len(df):,}")
    print(f"  Depth stats:   min={df['depth'].min()}, max={df['depth'].max()}, "
          f"mean={df['depth'].mean():.2f}")

    # Aggregate stats
    print("\n[2/5] Computing structural statistics...")
    struct_stats = compute_structural_stats(df)

    print("\n  Dependency rate by sameFile:")
    print(struct_stats["by_sameFile"].to_string(index=False))
    print("\n  Dependency rate by file_distance (depth):")
    print(struct_stats["by_distance"].to_string(index=False))

    # Save CSV
    print("\n[3/5] Saving analysis CSV...")
    export_cols = [
        "revision", "gcid1", "gcid2", "classId", "sameFile", "same_folder",
        "file_distance", "distance_bucket", "dep_label", "depth", "similarity",
        "filepath1", "filepath2", "folder1", "folder2",
    ]
    available = [c for c in export_cols if c in df.columns]
    csv_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_file-folder-analysis.csv")
    df[available].to_csv(csv_path, index=False)
    print(f"  [OK] Saved {csv_path}")

    # Save gcid-to-file lookup table
    print("\n  Generating gcid-to-file lookup table...")
    rev_frag = pd.read_csv(REV_FRAGMENT)
    gcid_file = rev_frag.groupby("gcid").agg(
        filePath=("filePath", "last"),
        folder=("filePath", lambda x: os.path.dirname(str(x.iloc[-1])).replace("\\", "/")),
        all_files=("filePath", lambda x: "; ".join(sorted(x.unique()))),
        num_files=("filePath", "nunique"),
        total_changes=("changeType", lambda x: (x.str.upper() == "M").sum()),
        first_revision=("revision", "min"),
        last_revision=("revision", "max"),
    ).reset_index()
    gcid_file_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_gcid_file_lookup.csv")
    gcid_file.to_csv(gcid_file_path, index=False)
    print(f"  [OK] Saved {gcid_file_path}")
    print(f"  {len(gcid_file)} gcids mapped to files")

    # Statistical tests
    print("\n[4/5] Running statistical tests...")
    test_results = run_statistical_tests(df)

    stats_lines = []
    stats_lines.append(f"File/Folder Structural Proximity - {SYSTEM} ({CLONE_TYPE})")
    stats_lines.append("=" * 60)
    stats_lines.append("")

    stats_lines.append("1. Same-File x Dependency (chi2 / Fisher)")
    for key in ["sameFile_chi2", "sameFile_chi2_pvalue", "sameFile_cramersV",
                 "sameFile_fisher_OR", "sameFile_fisher_pvalue"]:
        if key in test_results:
            val = test_results[key]
            stats_lines.append(f"   {key}: {val:.4f}" if isinstance(val, float)
                               else f"   {key}: {val}")
    stats_lines.append("")

    stats_lines.append("2. Mann-Whitney U: file_distance (dep vs. indep)")
    for key in ["mannwhitney_U", "mannwhitney_pvalue",
                 "dep_distance_median", "dep_distance_mean",
                 "indep_distance_median", "indep_distance_mean"]:
        if key in test_results:
            val = test_results[key]
            stats_lines.append(f"   {key}: {val:.4f}" if isinstance(val, float)
                               else f"   {key}: {val}")
    stats_lines.append("")

    stats_lines.append("3. Point-biserial: file_distance <-> dep_label")
    for key in ["pointbiserial_r", "pointbiserial_pvalue"]:
        if key in test_results:
            stats_lines.append(f"   {key}: {test_results[key]:.4f}")
    stats_lines.append("")

    if "odds_ratios" in test_results:
        stats_lines.append("4. Logistic Regression: dep ~ sameFile + file_distance")
        stats_lines.append(f"   Accuracy: {test_results['logistic_accuracy']:.4f}")
        for feat, or_val in test_results["odds_ratios"].items():
            stats_lines.append(f"   OR({feat}): {or_val:.4f}")

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    stats_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_filefolder_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\n  [OK] Saved {stats_path}")

    # Plots
    print("\n[5/5] Generating plots...")
    plot_distance_bars(struct_stats, RESULTS_DIR)
    plot_samefile_bars(struct_stats, test_results, RESULTS_DIR)
    plot_distance_violin(df, test_results, RESULTS_DIR)
    plot_cross_heatmap(struct_stats, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("  DONE - File/folder analysis complete for", SYSTEM)
    print("=" * 60)


if __name__ == "__main__":
    main()
