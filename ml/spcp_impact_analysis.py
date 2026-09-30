#!/usr/bin/env python3
"""
SPCP (Similarity-Preserving Co-evolving Pair) Impact Analysis
=============================================================
Standalone interpretability analysis that investigates whether
SPCP membership, coupling strength, and dominantChangeCategory
drive dependent vs. independent change behaviour.

Mirrors the structure of ``file_folder_analysis.py`` but swaps the
structural axis from file/folder proximity to SPCP attributes.

Data sources:
  - rev_pair.csv          (per-pair-revision events with isSpcp,
                           weightedCouplingStrength, sameFile, depth)
  - SPCP_<CloneType>.csv  (pair-level couplingStrength,
                           dominantChangeCategory, latePropagationCount)

Outputs (in ml/results/spcp_impact_analysis/<SYSTEM>/):
  - <SYSTEM>_spcp-impact-analysis.csv
  - <SYSTEM>_spcp_isspcp.png
  - <SYSTEM>_spcp_coupling.png
  - <SYSTEM>_spcp_coupling_violin.png
  - <SYSTEM>_spcp_dominantcategory.png
  - <SYSTEM>_spcp_isspcp_samefile_heatmap.png
  - <SYSTEM>_spcp_coupling_samefile_heatmap.png
  - <SYSTEM>_spcp_stats.txt
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
SYSTEM = "jEdit"
CLONE_TYPE = "Type3_Block"
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "WorkFolder", SYSTEM,
)
REV_PAIR = os.path.join(BASE_DIR, "Datasets", "CloneGenealogy", f"{CLONE_TYPE}_rev_pair.csv")
SPCP_FILE = os.path.join(BASE_DIR, f"SPCP_{CLONE_TYPE}.csv")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "spcp_impact_analysis", SYSTEM,
)
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
SPCP_COLOR = "#3498db"
NONSPCP_COLOR = "#95a5a6"

COUPLING_BINS = [-0.001, 0.0, 0.25, 0.5, 0.75, 1.01]
COUPLING_LABELS = ["0", "(0, 0.25]", "(0.25, 0.5]", "(0.5, 0.75]", "(0.75, 1]"]


# ──────────────────────────────────────────────────────────────────────
# 1. Load & enrich data
# ──────────────────────────────────────────────────────────────────────
def load_data():
    """Load rev_pair and enrich with SPCP pair-level attributes."""
    rev_pair = pd.read_csv(REV_PAIR)
    spcp = pd.read_csv(SPCP_FILE)

    # Dep label: both sides changed (M/A/D — anything but U)
    rev_pair["dep_label"] = (
        (rev_pair["changeType1"] != "U") & (rev_pair["changeType2"] != "U")
    ).astype(int)

    # Pair-level enrichment from SPCP file
    spcp_slim = spcp[[
        "gcid1", "gcid2", "couplingStrength",
        "dominantChangeCategory", "latePropagationCount",
        "no_of_revision_paired",
    ]].rename(columns={
        "couplingStrength": "pair_couplingStrength",
        "latePropagationCount": "pair_latePropagationCount",
        "no_of_revision_paired": "pair_no_of_revision_paired",
    })

    merged = rev_pair.merge(spcp_slim, on=["gcid1", "gcid2"], how="left")

    # Bucket weighted coupling strength (per-event)
    merged["coupling_bucket"] = pd.cut(
        merged["weightedCouplingStrength"].fillna(0.0),
        bins=COUPLING_BINS,
        labels=COUPLING_LABELS,
        include_lowest=True,
    )

    # Fill category for non-SPCP pairs so we can plot side-by-side
    merged["dominantChangeCategory"] = merged["dominantChangeCategory"].fillna("NON_SPCP")

    return merged


# ──────────────────────────────────────────────────────────────────────
# 2. Aggregate statistics
# ──────────────────────────────────────────────────────────────────────
def compute_structural_stats(df):
    results = {}

    # --- By isSpcp ---
    sp = df.groupby("isSpcp").agg(
        total=("dep_label", "count"),
        dependent=("dep_label", "sum"),
    ).reset_index()
    sp["independent"] = sp["total"] - sp["dependent"]
    sp["dep_rate"] = sp["dependent"] / sp["total"]
    results["by_isSpcp"] = sp

    # --- By coupling bucket ---
    cb = df.groupby("coupling_bucket", observed=False).agg(
        total=("dep_label", "count"),
        dependent=("dep_label", "sum"),
    ).reset_index()
    cb["independent"] = cb["total"] - cb["dependent"]
    cb["dep_rate"] = np.where(cb["total"] > 0, cb["dependent"] / cb["total"], 0)
    results["by_coupling"] = cb

    # --- By dominantChangeCategory (includes NON_SPCP bucket) ---
    dc = df.groupby("dominantChangeCategory").agg(
        total=("dep_label", "count"),
        dependent=("dep_label", "sum"),
    ).reset_index()
    dc["independent"] = dc["total"] - dc["dependent"]
    dc["dep_rate"] = dc["dependent"] / dc["total"]
    dc = dc.sort_values("dep_rate", ascending=False).reset_index(drop=True)
    results["by_dominantCategory"] = dc

    # --- isSpcp x sameFile interaction ---
    if "sameFile" in df.columns:
        cross_sf = df.groupby(["isSpcp", "sameFile"]).agg(
            total=("dep_label", "count"),
            dependent=("dep_label", "sum"),
        ).reset_index()
        cross_sf["dep_rate"] = np.where(
            cross_sf["total"] > 0, cross_sf["dependent"] / cross_sf["total"], 0
        )
        results["cross_isSpcp_sameFile"] = cross_sf

        # --- coupling x sameFile interaction ---
        cross_cb = df.groupby(["coupling_bucket", "sameFile"], observed=False).agg(
            total=("dep_label", "count"),
            dependent=("dep_label", "sum"),
        ).reset_index()
        cross_cb["dep_rate"] = np.where(
            cross_cb["total"] > 0, cross_cb["dependent"] / cross_cb["total"], 0
        )
        results["cross_coupling_sameFile"] = cross_cb

    return results


# ──────────────────────────────────────────────────────────────────────
# 3. Statistical tests
# ──────────────────────────────────────────────────────────────────────
def run_statistical_tests(df):
    results = {}

    # --- 3a. Chi-squared / Fisher: isSpcp x dep_label ---
    ct = pd.crosstab(df["isSpcp"], df["dep_label"])
    results["isSpcp_contingency"] = ct
    if ct.shape == (2, 2):
        chi2, p_chi2, _, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        cramers_v = np.sqrt(chi2 / n) if n > 0 else 0
        results["isSpcp_chi2"] = chi2
        results["isSpcp_chi2_pvalue"] = p_chi2
        results["isSpcp_cramersV"] = cramers_v
        odds, p_fisher = stats.fisher_exact(ct.values)
        results["isSpcp_fisher_OR"] = odds
        results["isSpcp_fisher_pvalue"] = p_fisher

    # --- 3b. Mann-Whitney U: weightedCouplingStrength for dep vs. indep ---
    valid = df.dropna(subset=["weightedCouplingStrength"])
    dep_cs = valid.loc[valid["dep_label"] == 1, "weightedCouplingStrength"]
    indep_cs = valid.loc[valid["dep_label"] == 0, "weightedCouplingStrength"]
    if len(dep_cs) > 0 and len(indep_cs) > 0:
        u_stat, p_mw = stats.mannwhitneyu(dep_cs, indep_cs, alternative="two-sided")
        results["mannwhitney_U"] = u_stat
        results["mannwhitney_pvalue"] = p_mw
        results["dep_coupling_median"] = dep_cs.median()
        results["dep_coupling_mean"] = dep_cs.mean()
        results["indep_coupling_median"] = indep_cs.median()
        results["indep_coupling_mean"] = indep_cs.mean()

    # --- 3c. Point-biserial: weightedCouplingStrength <-> dep_label ---
    if len(valid) > 2:
        r_pb, p_pb = stats.pointbiserialr(valid["dep_label"], valid["weightedCouplingStrength"])
        results["pointbiserial_r"] = r_pb
        results["pointbiserial_pvalue"] = p_pb

    # --- 3d. Kruskal-Wallis across dominantChangeCategory groups ---
    cats = df["dominantChangeCategory"].dropna().unique()
    groups = [df.loc[df["dominantChangeCategory"] == c, "dep_label"].values for c in cats]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        kw_stat, p_kw = stats.kruskal(*groups)
        results["kruskal_H"] = kw_stat
        results["kruskal_pvalue"] = p_kw
        results["kruskal_n_groups"] = len(groups)

    # --- 3e. Logistic regression: dep ~ isSpcp + weightedCouplingStrength + sameFile + depth ---
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        feat_cols = ["isSpcp", "weightedCouplingStrength"]
        if "sameFile" in df.columns:
            feat_cols.append("sameFile")
        if "depth" in df.columns:
            feat_cols.append("depth")

        features = df[feat_cols].copy()
        features["weightedCouplingStrength"] = features["weightedCouplingStrength"].fillna(0)
        if "depth" in features.columns:
            features["depth"] = features["depth"].fillna(0)
        features = features.dropna()
        y = df.loc[features.index, "dep_label"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)
        results["logistic_coefficients"] = dict(zip(features.columns, lr.coef_[0]))
        results["logistic_intercept"] = lr.intercept_[0]
        results["logistic_accuracy"] = lr.score(X_scaled, y)

        lr2 = LogisticRegression(max_iter=1000, random_state=42)
        lr2.fit(features, y)
        results["odds_ratios"] = {
            k: float(np.exp(v)) for k, v in zip(features.columns, lr2.coef_[0])
        }
    except ImportError:
        pass

    return results


# ──────────────────────────────────────────────────────────────────────
# 4. Visualisations
# ──────────────────────────────────────────────────────────────────────
def plot_isspcp_bars(struct_stats, test_results, out_dir):
    sp = struct_stats["by_isSpcp"]
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Non-SPCP", "SPCP"]
    colors = [NONSPCP_COLOR, SPCP_COLOR]

    for i, (_, row) in enumerate(sp.iterrows()):
        ax.bar(i, row["dep_rate"], 0.5, color=colors[i], edgecolor="white", linewidth=0.5)
        ax.text(i, row["dep_rate"] + 0.01,
                f'{row["dep_rate"]:.1%}\n(n={row["total"]:,})',
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    p_val = test_results.get("isSpcp_fisher_pvalue", test_results.get("isSpcp_chi2_pvalue"))
    or_val = test_results.get("isSpcp_fisher_OR")
    title = f"SPCP vs. Non-SPCP Dependency Rate - {SYSTEM}"
    if p_val is not None and or_val is not None:
        title += f"\nFisher OR={or_val:.3f}  |  p={p_val:.2e}"

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Dependency Rate")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(sp["dep_rate"].max() + 0.15, 1.05))
    ax.grid(axis="y", alpha=0.2)
    sns.despine()
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_spcp_isspcp.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_spcp_isspcp.png")


def plot_coupling_bars(struct_stats, out_dir):
    cb = struct_stats["by_coupling"]
    cb = cb[cb["total"] > 0].copy().reset_index(drop=True)
    if cb.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cb))
    ax.bar(x, cb["dep_rate"], 0.5, color=PALETTE[3], edgecolor="white", linewidth=0.5)

    for i, (_, row) in enumerate(cb.iterrows()):
        ax.text(i, row["dep_rate"] + 0.01,
                f'{row["dep_rate"]:.1%}\n(n={row["total"]:,})',
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(cb["coupling_bucket"])
    ax.set_xlabel("Weighted Coupling Strength (bucket)")
    ax.set_ylabel("Dependency Rate")
    ax.set_title(f"Dependency Rate by Coupling Strength - {SYSTEM} ({CLONE_TYPE})")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(cb["dep_rate"].max() + 0.15, 1.05))
    ax.grid(axis="y", alpha=0.2)
    sns.despine()
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_spcp_coupling.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_spcp_coupling.png")


def plot_coupling_violin(df, test_results, out_dir):
    valid = df.dropna(subset=["weightedCouplingStrength"]).copy()
    if valid.empty:
        return
    dep_vals = valid.loc[valid["dep_label"] == 1, "weightedCouplingStrength"].values
    indep_vals = valid.loc[valid["dep_label"] == 0, "weightedCouplingStrength"].values
    if len(dep_vals) == 0 or len(indep_vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    parts = ax.violinplot([dep_vals, indep_vals], positions=[0, 1],
                           showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([DEP_COLOR, INDEP_COLOR][i])
        pc.set_alpha(0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Dependent", "Independent"])
    ax.set_ylabel("Weighted Coupling Strength")

    p_mw = test_results.get("mannwhitney_pvalue")
    med_dep = test_results.get("dep_coupling_median")
    med_ind = test_results.get("indep_coupling_median")
    title = f"Coupling Strength Distribution - {SYSTEM}\n"
    if p_mw is not None:
        title += f"Mann-Whitney p={p_mw:.2e} | Median: dep={med_dep:.3f}, indep={med_ind:.3f}"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    sns.despine()
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_spcp_coupling_violin.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_spcp_coupling_violin.png")


def plot_dominant_category_bars(struct_stats, test_results, out_dir):
    dc = struct_stats["by_dominantCategory"]
    dc = dc[dc["total"] > 0].copy().reset_index(drop=True)
    if dc.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dc))
    colors = [NONSPCP_COLOR if c == "NON_SPCP" else SPCP_COLOR
              for c in dc["dominantChangeCategory"]]
    ax.bar(x, dc["dep_rate"], 0.6, color=colors, edgecolor="white", linewidth=0.5)

    for i, (_, row) in enumerate(dc.iterrows()):
        ax.text(i, row["dep_rate"] + 0.01,
                f'{row["dep_rate"]:.1%}\n(n={row["total"]:,})',
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(dc["dominantChangeCategory"], rotation=15, ha="right")
    ax.set_ylabel("Dependency Rate")

    p_kw = test_results.get("kruskal_pvalue")
    title = f"Dependency Rate by Dominant Change Category - {SYSTEM}"
    if p_kw is not None:
        title += f"\nKruskal-Wallis p={p_kw:.2e}"
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(dc["dep_rate"].max() + 0.15, 1.05))
    ax.grid(axis="y", alpha=0.2)
    sns.despine()
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_spcp_dominantcategory.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_spcp_dominantcategory.png")


def plot_isspcp_samefile_heatmap(struct_stats, out_dir):
    if "cross_isSpcp_sameFile" not in struct_stats:
        return
    cross = struct_stats["cross_isSpcp_sameFile"]
    cross = cross[cross["total"] > 0].copy()
    if cross.empty:
        return

    pivot = cross.pivot_table(index="isSpcp", columns="sameFile",
                               values="dep_rate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax,
                yticklabels=["Non-SPCP", "SPCP"],
                xticklabels=["Cross-file", "Same file"],
                linewidths=0.5)
    ax.set_xlabel("Same File")
    ax.set_ylabel("SPCP membership")
    ax.set_title(f"Dependency Rate: SPCP x Same-File - {SYSTEM}")
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_spcp_isspcp_samefile_heatmap.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_spcp_isspcp_samefile_heatmap.png")


def plot_coupling_samefile_heatmap(struct_stats, out_dir):
    if "cross_coupling_sameFile" not in struct_stats:
        return
    cross = struct_stats["cross_coupling_sameFile"]
    cross = cross[cross["total"] > 0].copy()
    if cross.empty:
        return

    pivot = cross.pivot_table(index="coupling_bucket", columns="sameFile",
                               values="dep_rate", aggfunc="mean", observed=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax,
                xticklabels=["Cross-file", "Same file"], linewidths=0.5)
    ax.set_xlabel("Same File")
    ax.set_ylabel("Coupling Strength bucket")
    ax.set_title(f"Dependency Rate: Coupling x Same-File - {SYSTEM}")
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_spcp_coupling_samefile_heatmap.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_spcp_coupling_samefile_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  SPCP Impact Analysis - {SYSTEM} ({CLONE_TYPE})")
    print("=" * 60 + "\n")

    print("[1/5] Loading and enriching data...")
    df = load_data()
    print(f"  Events:           {len(df):,}")
    print(f"  Dep events:       {df['dep_label'].sum():,} "
          f"({df['dep_label'].mean():.1%})")
    print(f"  isSpcp=1 events:  {(df['isSpcp']==1).sum():,} "
          f"({(df['isSpcp']==1).mean():.1%})")
    print(f"  Pair-enriched:    {df['pair_couplingStrength'].notna().sum():,} "
          f"events match SPCP file")
    print(f"  Coupling stats:   min={df['weightedCouplingStrength'].min():.3f}, "
          f"max={df['weightedCouplingStrength'].max():.3f}, "
          f"mean={df['weightedCouplingStrength'].mean():.3f}")

    print("\n[2/5] Computing structural statistics...")
    struct_stats = compute_structural_stats(df)
    print("\n  Dependency rate by isSpcp:")
    print(struct_stats["by_isSpcp"].to_string(index=False))
    print("\n  Dependency rate by coupling bucket:")
    print(struct_stats["by_coupling"].to_string(index=False))
    print("\n  Dependency rate by dominantChangeCategory:")
    print(struct_stats["by_dominantCategory"].to_string(index=False))

    print("\n[3/5] Saving analysis CSV...")
    export_cols = [
        "revision", "gcid1", "gcid2", "classId",
        "changeType1", "changeType2", "dep_label",
        "isSpcp", "weightedCouplingStrength", "coupling_bucket",
        "pair_couplingStrength", "dominantChangeCategory",
        "pair_latePropagationCount", "pair_no_of_revision_paired",
        "sameFile", "depth", "similarity",
    ]
    available = [c for c in export_cols if c in df.columns]
    csv_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_spcp-impact-analysis.csv")
    df[available].to_csv(csv_path, index=False)
    print(f"  [OK] Saved {csv_path}")

    print("\n[4/5] Running statistical tests...")
    test_results = run_statistical_tests(df)

    lines = []
    lines.append(f"SPCP Impact Analysis - {SYSTEM} ({CLONE_TYPE})")
    lines.append("=" * 60)
    lines.append("")
    lines.append("1. isSpcp x Dependency (chi2 / Fisher)")
    for key in ["isSpcp_chi2", "isSpcp_chi2_pvalue", "isSpcp_cramersV",
                 "isSpcp_fisher_OR", "isSpcp_fisher_pvalue"]:
        if key in test_results:
            v = test_results[key]
            lines.append(f"   {key}: {v:.4f}" if isinstance(v, float) else f"   {key}: {v}")
    lines.append("")
    lines.append("   Contingency (rows=isSpcp, cols=dep_label):")
    lines.append(str(test_results["isSpcp_contingency"]))
    lines.append("")

    lines.append("2. Mann-Whitney U: weightedCouplingStrength (dep vs. indep)")
    for key in ["mannwhitney_U", "mannwhitney_pvalue",
                 "dep_coupling_median", "dep_coupling_mean",
                 "indep_coupling_median", "indep_coupling_mean"]:
        if key in test_results:
            lines.append(f"   {key}: {test_results[key]:.4f}")
    lines.append("")

    lines.append("3. Point-biserial: weightedCouplingStrength <-> dep_label")
    for key in ["pointbiserial_r", "pointbiserial_pvalue"]:
        if key in test_results:
            lines.append(f"   {key}: {test_results[key]:.4f}")
    lines.append("")

    lines.append("4. Kruskal-Wallis across dominantChangeCategory groups")
    for key in ["kruskal_H", "kruskal_pvalue", "kruskal_n_groups"]:
        if key in test_results:
            v = test_results[key]
            lines.append(f"   {key}: {v:.4f}" if isinstance(v, float) else f"   {key}: {v}")
    lines.append("")

    if "odds_ratios" in test_results:
        lines.append("5. Logistic Regression: dep ~ isSpcp + coupling + sameFile + depth")
        lines.append(f"   Accuracy: {test_results['logistic_accuracy']:.4f}")
        for feat, or_val in test_results["odds_ratios"].items():
            lines.append(f"   OR({feat}): {or_val:.4f}")

    stats_text = "\n".join(lines)
    print(stats_text)
    stats_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_spcp_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\n  [OK] Saved {stats_path}")

    print("\n[5/5] Generating plots...")
    plot_isspcp_bars(struct_stats, test_results, RESULTS_DIR)
    plot_coupling_bars(struct_stats, RESULTS_DIR)
    plot_coupling_violin(df, test_results, RESULTS_DIR)
    plot_dominant_category_bars(struct_stats, test_results, RESULTS_DIR)
    plot_isspcp_samefile_heatmap(struct_stats, RESULTS_DIR)
    plot_coupling_samefile_heatmap(struct_stats, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("  DONE - SPCP impact analysis complete for", SYSTEM)
    print("=" * 60)


if __name__ == "__main__":
    main()
