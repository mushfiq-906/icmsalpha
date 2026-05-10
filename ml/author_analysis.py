#!/usr/bin/env python3
"""
Author-Based Dependency Triggering Analysis
============================================
Standalone interpretability analysis that investigates whether the
committer (author) behind a change is a significant driver of
dependent vs. independent clone fragment change behaviour.

Currently implemented for: Ctags (Type3_Block)

Data sources:
  - rev_fragment.csv  (per-fragment change events with author)
  - rev_pair.csv      (per-pair change events)

Outputs (in ml/results/):
  - Ctags_author-based-analysis.csv
  - Ctags_author_profile.png
  - Ctags_author_heatmap.png
  - Ctags_author_scatter.png
  - Ctags_author_boxplot.png
  - Ctags_author_stats.txt
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
SYSTEM = "dnsjava"
CLONE_TYPE = "Type3_Block"
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "WorkFolder", SYSTEM, "Datasets", "CloneGenealogy"
)
REV_FRAGMENT = os.path.join(BASE_DIR, f"{CLONE_TYPE}_rev_fragment.csv")
REV_PAIR = os.path.join(BASE_DIR, f"{CLONE_TYPE}_rev_pair.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "author_analysis", SYSTEM)
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
# 1. Load & label data
# ──────────────────────────────────────────────────────────────────────
def load_data():
    """Load rev_fragment and rev_pair, derive dependency labels."""
    frag = pd.read_csv(REV_FRAGMENT)
    pair = pd.read_csv(REV_PAIR)

    # --- Pair-level: label each pair-revision as dependent or independent ---
    # dependent  = both sides changed (changeType1 != 'U' AND changeType2 != 'U')
    # independent = only one side changed
    pair["dep_label"] = (
        (pair["changeType1"] != "U") & (pair["changeType2"] != "U")
    ).astype(int)

    return frag, pair


def build_author_pair_events(frag, pair):
    """
    For each pair-revision event, determine the author(s) involved
    by joining with fragment data on both sides.

    Also DERIVES a proper same_author flag from actual author names
    (the rev_pair.csv sameAuthor column is always 1 in Ctags).
    """
    # Get the author for each (revision, gcid) from fragment data
    author_map = frag[["revision", "gcid", "author"]].drop_duplicates()

    # Join author for gcid1
    merged = pair.merge(
        author_map.rename(columns={"gcid": "gcid1", "author": "author1"}),
        on=["revision", "gcid1"],
        how="left",
    )
    # Join author for gcid2
    merged = merged.merge(
        author_map.rename(columns={"gcid": "gcid2", "author": "author2"}),
        on=["revision", "gcid2"],
        how="left",
    )

    # Derive proper same_author from actual names
    # For dep events: both sides have authors -> compare them
    # For indep events: only one side has author -> mark as "N/A" (single-side)
    merged["derived_same_author"] = np.nan  # default

    # Both changed -> we can compare
    both_changed = (merged["changeType1"] != "U") & (merged["changeType2"] != "U")
    has_both = both_changed & merged["author1"].notna() & merged["author2"].notna()
    merged.loc[has_both, "derived_same_author"] = (
        merged.loc[has_both, "author1"] == merged.loc[has_both, "author2"]
    ).astype(int)

    return merged


# ──────────────────────────────────────────────────────────────────────
# 2. Per-author statistics
# ──────────────────────────────────────────────────────────────────────
def compute_author_stats(merged):
    """Compute per-author dependency statistics from pair-level events."""
    records = []

    for _, row in merged.iterrows():
        dep = row["dep_label"]
        same_auth = row["derived_same_author"]

        # Collect active authors (those whose side actually changed)
        active = []
        if row["changeType1"] != "U" and pd.notna(row.get("author1")):
            active.append(row["author1"])
        if row["changeType2"] != "U" and pd.notna(row.get("author2")):
            active.append(row["author2"])

        for author in set(active):
            records.append({
                "author": author,
                "dep_label": dep,
                "derived_same_author": same_auth,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    # Aggregate per author
    agg = df.groupby("author").agg(
        total_changes=("dep_label", "count"),
        dependent_count=("dep_label", "sum"),
    ).reset_index()
    agg["independent_count"] = agg["total_changes"] - agg["dependent_count"]
    agg["dep_ratio"] = agg["dependent_count"] / agg["total_changes"]

    # Same-author breakdowns (only for dep events where derived_same_author is known)
    for auth_name in agg["author"].unique():
        subset = df[df["author"] == auth_name]
        valid = subset[subset["derived_same_author"].notna()]

        sa_dep = ((valid["derived_same_author"] == 1) & (valid["dep_label"] == 1)).sum()
        da_dep = ((valid["derived_same_author"] == 0) & (valid["dep_label"] == 1)).sum()
        sa_ind = ((valid["derived_same_author"] == 1) & (valid["dep_label"] == 0)).sum()
        da_ind = ((valid["derived_same_author"] == 0) & (valid["dep_label"] == 0)).sum()

        idx = agg["author"] == auth_name
        agg.loc[idx, "same_author_dep_count"] = sa_dep
        agg.loc[idx, "diff_author_dep_count"] = da_dep
        agg.loc[idx, "same_author_indep_count"] = sa_ind
        agg.loc[idx, "diff_author_indep_count"] = da_ind

    # Co-change trigger rate
    same_total = agg["same_author_dep_count"] + agg["same_author_indep_count"]
    agg["co_change_trigger_rate"] = np.where(
        same_total > 0, agg["same_author_dep_count"] / same_total, np.nan
    )
    agg["solo_change_rate"] = agg["independent_count"] / agg["total_changes"]

    int_cols = [
        "same_author_dep_count", "diff_author_dep_count",
        "same_author_indep_count", "diff_author_indep_count",
    ]
    agg[int_cols] = agg[int_cols].astype(int)

    return agg.sort_values("total_changes", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# 3. Statistical tests
# ──────────────────────────────────────────────────────────────────────
def run_statistical_tests(merged, author_stats):
    """Run chi-squared, Fisher's exact, and Cramer's V."""
    results = {}

    # --- 3a. Global: derived_same_author x dep_label (2x2 contingency) ---
    # Only use events where derived_same_author is known (dep events with both authors)
    valid = merged[merged["derived_same_author"].notna()].copy()
    valid["derived_same_author"] = valid["derived_same_author"].astype(int)

    if len(valid) > 0 and valid["derived_same_author"].nunique() > 1:
        ct = pd.crosstab(valid["derived_same_author"], valid["dep_label"])
        chi2, p_chi2, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        k = min(ct.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1))) if n > 0 else 0

        results["sameAuthor_x_dep_chi2"] = chi2
        results["sameAuthor_x_dep_pvalue"] = p_chi2
        results["sameAuthor_x_dep_dof"] = dof
        results["sameAuthor_x_dep_cramersV"] = cramers_v
        results["contingency_table"] = ct

        # Fisher's exact (2x2)
        if ct.shape == (2, 2):
            odds, p_fisher = stats.fisher_exact(ct.values)
            results["fisher_odds_ratio"] = odds
            results["fisher_pvalue"] = p_fisher
    else:
        # All dep events have same author (e.g. single-author project)
        results["note_same_author"] = (
            f"derived_same_author has {valid['derived_same_author'].nunique()} "
            f"unique value(s) across {len(valid)} dep events. "
            "Chi-squared not applicable (no variation)."
        )
        if len(valid) > 0:
            ct = pd.crosstab(valid["derived_same_author"], valid["dep_label"])
            results["contingency_table"] = ct

    # --- 3b. Author identity x dep_label (multi-category chi-squared) ---
    top = author_stats[author_stats["total_changes"] >= 5].copy()
    if len(top) >= 2:
        auth_ct = top[["author", "dependent_count", "independent_count"]].set_index(
            "author"
        )
        chi2_auth, p_auth, dof_auth, _ = stats.chi2_contingency(auth_ct.values)
        n_auth = auth_ct.values.sum()
        k_auth = min(auth_ct.shape)
        cramers_v_auth = np.sqrt(chi2_auth / (n_auth * (k_auth - 1))) if n_auth > 0 else 0
        results["author_identity_chi2"] = chi2_auth
        results["author_identity_pvalue"] = p_auth
        results["author_identity_dof"] = dof_auth
        results["author_identity_cramersV"] = cramers_v_auth

    return results


# ──────────────────────────────────────────────────────────────────────
# 4. Visualisations
# ──────────────────────────────────────────────────────────────────────
def plot_author_profile(author_stats, out_dir):
    """Bar chart: top 15 authors, stacked dep vs indep."""
    top = author_stats.head(15).copy().reset_index(drop=True)
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top))
    w = 0.6

    ax.bar(x, top["dependent_count"], w, label="Dependent (co-change)",
           color=DEP_COLOR, edgecolor="white", linewidth=0.5)
    ax.bar(x, top["independent_count"], w, bottom=top["dependent_count"],
           label="Independent (solo)", color=INDEP_COLOR, edgecolor="white",
           linewidth=0.5)

    # Annotate dep_ratio on top
    for i, row in top.iterrows():
        total = row["dependent_count"] + row["independent_count"]
        ax.text(
            x[i], total + 1,
            f"{row['dep_ratio']:.0%}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(top["author"], rotation=45, ha="right")
    ax.set_ylabel("Number of Change Events")
    ax.set_title(f"Author Dependency Profile - {SYSTEM} ({CLONE_TYPE})\n"
                 f"Top {len(top)} authors by volume, annotated with dependency ratio")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_author_profile.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_author_profile.png")


def plot_same_author_heatmap(merged, test_results, out_dir):
    """2x2 heatmap: derived_same_author x dep_label with counts and rates."""
    ct = test_results.get("contingency_table")
    if ct is None or ct.shape != (2, 2):
        # If we don't have a proper 2x2, make a summary heatmap anyway
        valid = merged[merged["derived_same_author"].notna()].copy()
        if len(valid) == 0:
            print("  [SKIP] No valid same-author data for heatmap")
            return
        valid["derived_same_author"] = valid["derived_same_author"].astype(int)
        ct = pd.crosstab(valid["derived_same_author"], valid["dep_label"])

    # Normalise for annotation
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Row labels
    row_labels = []
    for idx in ct.index:
        row_labels.append("Same Author" if idx == 1 else "Diff Author")
    col_labels = ["Independent", "Dependent"]

    # Counts
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=axes[0],
                xticklabels=col_labels[:len(ct.columns)],
                yticklabels=row_labels)
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Change Type")
    axes[0].set_ylabel("Author Overlap")

    # Rates
    sns.heatmap(ct_norm, annot=True, fmt=".1%", cmap="YlGnBu", ax=axes[1],
                xticklabels=col_labels[:len(ct_norm.columns)],
                yticklabels=row_labels)
    axes[1].set_title("Row-normalised Rates")
    axes[1].set_xlabel("Change Type")
    axes[1].set_ylabel("Author Overlap")

    p_val = test_results.get("fisher_pvalue", test_results.get("sameAuthor_x_dep_pvalue"))
    cramers = test_results.get("sameAuthor_x_dep_cramersV", 0)
    note = test_results.get("note_same_author", "")

    if p_val is not None:
        fig.suptitle(
            f"Same-Author vs. Different-Author - {SYSTEM}\n"
            f"Chi2 p={p_val:.2e}  |  Cramer's V={cramers:.4f}",
            fontsize=13, fontweight="bold",
        )
    else:
        fig.suptitle(
            f"Same-Author vs. Different-Author - {SYSTEM}\n{note}",
            fontsize=12, fontweight="bold",
        )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_author_heatmap.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_author_heatmap.png")


def plot_author_scatter(author_stats, out_dir):
    """Scatter: X=total events, Y=dep_ratio, one dot per author."""
    df = author_stats[author_stats["total_changes"] >= 3].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df["total_changes"], df["dep_ratio"],
        s=60, alpha=0.7, c=df["dep_ratio"],
        cmap="RdYlGn", edgecolors="grey", linewidth=0.5,
        vmin=0, vmax=1,
    )
    plt.colorbar(scatter, ax=ax, label="Dependency Ratio", shrink=0.8)

    # Label all authors (small dataset)
    for _, row in df.iterrows():
        ax.annotate(
            row["author"],
            (row["total_changes"], row["dep_ratio"]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=8, alpha=0.8,
        )

    ax.set_xlabel("Total Change Events")
    ax.set_ylabel("Dependency Ratio")
    ax.set_title(f"Author Consistency - {SYSTEM} ({CLONE_TYPE})\n"
                 f"Each dot = one author (>= 3 events)")
    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(alpha=0.2)
    sns.despine()

    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_author_scatter.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_author_scatter.png")


def plot_author_boxplot(author_stats, out_dir):
    """Box plot: distribution of dep_ratio across authors."""
    df = author_stats[author_stats["total_changes"] >= 3].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        df["dep_ratio"].dropna(), vert=True, patch_artist=True,
        boxprops=dict(facecolor=PALETTE[2], alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="grey"),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    # Overlay individual points (jittered)
    jitter = np.random.normal(1, 0.04, size=len(df))
    ax.scatter(jitter, df["dep_ratio"], alpha=0.5, s=20, color=PALETTE[4], zorder=5)

    ax.set_ylabel("Dependency Ratio")
    ax.set_title(
        f"Distribution of Author Dep. Ratios - {SYSTEM}\n"
        f"n={len(df)} authors (>= 3 events) | "
        f"median={df['dep_ratio'].median():.2%}"
    )
    ax.set_xticks([1])
    ax.set_xticklabels([SYSTEM])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.2)
    sns.despine()

    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_author_boxplot.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_author_boxplot.png")


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  Author-Based Dependency Analysis - {SYSTEM} ({CLONE_TYPE})")
    print("=" * 60 + "\n")

    # Load
    print("[1/5] Loading data...")
    frag, pair = load_data()
    print(f"  rev_fragment: {len(frag):,} rows")
    print(f"  rev_pair:     {len(pair):,} rows")
    print(f"  Dep events:   {pair['dep_label'].sum():,} / {len(pair):,} "
          f"({pair['dep_label'].mean():.1%})")

    # Build author-pair events
    print("\n[2/5] Joining author info with pair events...")
    merged = build_author_pair_events(frag, pair)
    n_with_a1 = merged["author1"].notna().sum()
    n_with_a2 = merged["author2"].notna().sum()
    n_derived = merged["derived_same_author"].notna().sum()
    print(f"  Merged rows:        {len(merged):,}")
    print(f"  author1 resolved:   {n_with_a1:,}")
    print(f"  author2 resolved:   {n_with_a2:,}")
    print(f"  derived_same_author available: {n_derived:,}")
    if n_derived > 0:
        v = merged["derived_same_author"].dropna()
        print(f"    Same author:      {int(v.sum()):,} ({v.mean():.1%})")
        print(f"    Diff author:      {int(len(v) - v.sum()):,}")

    # Per-author stats
    print("\n[3/5] Computing per-author statistics...")
    author_stats = compute_author_stats(merged)
    print(f"  Unique authors: {len(author_stats)}")
    print(f"\n  Top 10 authors by volume:")
    display_cols = ["author", "total_changes", "dependent_count",
                    "independent_count", "dep_ratio"]
    print(author_stats[display_cols].head(10).to_string(index=False))

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_author-based-analysis.csv")
    author_stats.to_csv(csv_path, index=False)
    print(f"\n  [OK] Saved {csv_path}")

    # Save gcid-to-author lookup table
    print("\n  Generating gcid-to-author lookup table...")
    gcid_author = frag.groupby("gcid").agg(
        all_authors=("author", lambda x: "; ".join(sorted(x.unique()))),
        primary_author=("author", lambda x: x.value_counts().index[0]),
        num_authors=("author", "nunique"),
        total_changes=("changeType", lambda x: (x.str.upper() == "M").sum()),
        first_revision=("revision", "min"),
        last_revision=("revision", "max"),
    ).reset_index()
    gcid_author_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_gcid_author_lookup.csv")
    gcid_author.to_csv(gcid_author_path, index=False)
    print(f"  [OK] Saved {gcid_author_path}")
    print(f"  {len(gcid_author)} gcids mapped to authors")

    # Statistical tests
    print("\n[4/5] Running statistical tests...")
    test_results = run_statistical_tests(merged, author_stats)

    stats_lines = []
    stats_lines.append(f"Author-Based Dependency Analysis - {SYSTEM} ({CLONE_TYPE})")
    stats_lines.append("=" * 60)
    stats_lines.append("")

    if "note_same_author" in test_results:
        stats_lines.append("1. Same-Author x Dependency")
        stats_lines.append(f"   NOTE: {test_results['note_same_author']}")
    else:
        stats_lines.append("1. Same-Author x Dependency (2x2 contingency)")
        stats_lines.append(f"   Chi-squared: {test_results.get('sameAuthor_x_dep_chi2', 'N/A'):.4f}")
        stats_lines.append(f"   p-value:     {test_results.get('sameAuthor_x_dep_pvalue', 'N/A'):.2e}")
        stats_lines.append(f"   Cramer's V:  {test_results.get('sameAuthor_x_dep_cramersV', 'N/A'):.4f}")
        if "fisher_odds_ratio" in test_results:
            stats_lines.append(f"   Fisher OR:   {test_results['fisher_odds_ratio']:.4f}")
            stats_lines.append(f"   Fisher p:    {test_results['fisher_pvalue']:.2e}")
    stats_lines.append("")

    if "author_identity_chi2" in test_results:
        stats_lines.append("2. Author Identity x Dependency (multi-category)")
        stats_lines.append(f"   Chi-squared: {test_results['author_identity_chi2']:.4f}")
        stats_lines.append(f"   p-value:     {test_results['author_identity_pvalue']:.2e}")
        stats_lines.append(f"   dof:         {test_results['author_identity_dof']}")
        stats_lines.append(f"   Cramer's V:  {test_results['author_identity_cramersV']:.4f}")

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    stats_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_author_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\n  [OK] Saved {stats_path}")

    # Plots
    print("\n[5/5] Generating plots...")
    plot_author_profile(author_stats, RESULTS_DIR)
    plot_same_author_heatmap(merged, test_results, RESULTS_DIR)
    plot_author_scatter(author_stats, RESULTS_DIR)
    plot_author_boxplot(author_stats, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("  DONE - Author analysis complete for", SYSTEM)
    print("=" * 60)


if __name__ == "__main__":
    main()
