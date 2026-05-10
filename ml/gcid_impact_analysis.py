#!/usr/bin/env python3
"""
gcid-Level Impact Analysis
===========================
Standalone interpretability analysis that investigates what makes
a specific clone fragment (gcid) change independently vs. dependently.

Combines author and file-location factors at the gcid level to show
which factor dominates and how they interact.

Currently implemented for: Ctags (Type3_Block)

Data sources:
  - rev_fragment.csv  (per-fragment change events)
  - rev_pair.csv      (per-pair change events)

Outputs (in ml/results/gcid_impact_analysis/{SYSTEM}/):
  - {SYSTEM}_gcid_impact.csv
  - {SYSTEM}_gcid_impact_stats.txt
  - {SYSTEM}_gcid_behavior_pie.png
  - {SYSTEM}_gcid_author_impact.png
  - {SYSTEM}_gcid_file_impact.png
  - {SYSTEM}_gcid_combined_heatmap.png
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
SYSTEM = "dnsjava"
CLONE_TYPE = "Type3_Block"
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "WorkFolder", SYSTEM, "Datasets", "CloneGenealogy"
)
REV_FRAGMENT = os.path.join(BASE_DIR, f"{CLONE_TYPE}_rev_fragment.csv")
REV_PAIR = os.path.join(BASE_DIR, f"{CLONE_TYPE}_rev_pair.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "gcid_impact_analysis", SYSTEM)
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
MIX_COLOR = "#f39c12"
INDEP_COLOR = "#e74c3c"


# ──────────────────────────────────────────────────────────────────────
# 1. Build gcid-level dataset
# ──────────────────────────────────────────────────────────────────────
def build_gcid_dataset():
    """Build a per-gcid dataset with author, file, and dependency stats."""
    frag = pd.read_csv(REV_FRAGMENT)
    pair = pd.read_csv(REV_PAIR)

    # Label pair events
    pair["dep_label"] = (
        (pair["changeType1"] != "U") & (pair["changeType2"] != "U")
    ).astype(int)

    records = []
    all_gcids = frag["gcid"].unique()
    print(f"  Processing {len(all_gcids)} gcids...")

    for gcid in all_gcids:
        mask = (pair["gcid1"] == gcid) | (pair["gcid2"] == gcid)
        sub = pair[mask]
        if len(sub) == 0:
            continue

        frag_sub = frag[frag["gcid"] == gcid]
        author = frag_sub["author"].value_counts().index[0]
        num_authors = frag_sub["author"].nunique()
        filepath = frag_sub["filePath"].iloc[-1]
        total_changes = int((frag_sub["changeType"].str.upper() == "M").sum())

        dep_events = int(sub["dep_label"].sum())
        indep_events = len(sub) - dep_events
        dep_ratio = dep_events / len(sub)
        same_file_frac = float(sub["sameFile"].mean())
        mean_depth = float(sub["depth"].mean())

        records.append({
            "gcid": gcid,
            "author": author,
            "num_authors": num_authors,
            "filePath": filepath,
            "total_own_changes": total_changes,
            "pair_events": len(sub),
            "dep_events": dep_events,
            "indep_events": indep_events,
            "dep_ratio": round(dep_ratio, 6),
            "same_file_frac": round(same_file_frac, 6),
            "mean_depth": round(mean_depth, 4),
        })

    df = pd.DataFrame(records)
    df["behavior"] = pd.cut(
        df["dep_ratio"],
        bins=[-0.01, 0.1, 0.5, 1.01],
        labels=["Mostly Independent", "Mixed", "Mostly Dependent"],
    )
    return df, frag, pair


# ──────────────────────────────────────────────────────────────────────
# 2. Statistical tests
# ──────────────────────────────────────────────────────────────────────
def run_gcid_tests(df):
    """Run statistical tests at the gcid level."""
    results = {}

    # --- Pearson: same_file_frac <-> dep_ratio ---
    r, p = stats.pearsonr(df["same_file_frac"], df["dep_ratio"])
    results["pearson_samefile_dep_r"] = r
    results["pearson_samefile_dep_p"] = p

    # --- Mann-Whitney: dep_ratio for gcids with vs without same-file pairs ---
    has_sf = df[df["same_file_frac"] > 0]["dep_ratio"]
    no_sf = df[df["same_file_frac"] == 0]["dep_ratio"]
    if len(has_sf) > 0 and len(no_sf) > 0:
        u, p_mw = stats.mannwhitneyu(has_sf, no_sf, alternative="two-sided")
        results["mannwhitney_file_U"] = u
        results["mannwhitney_file_p"] = p_mw
        results["has_samefile_mean_dr"] = float(has_sf.mean())
        results["no_samefile_mean_dr"] = float(no_sf.mean())
        results["has_samefile_n"] = len(has_sf)
        results["no_samefile_n"] = len(no_sf)

    # --- ANOVA: dep_ratio across authors ---
    groups = [g["dep_ratio"].values for _, g in df.groupby("author") if len(g) >= 3]
    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
        results["anova_author_F"] = f_stat
        results["anova_author_p"] = p_anova

    # --- Point-biserial: has_same_file (binary) <-> dep_ratio ---
    df_temp = df.copy()
    df_temp["has_sf_binary"] = (df_temp["same_file_frac"] > 0).astype(int)
    r_pb, p_pb = stats.pointbiserialr(df_temp["has_sf_binary"], df_temp["dep_ratio"])
    results["pointbiserial_hasfile_r"] = r_pb
    results["pointbiserial_hasfile_p"] = p_pb

    return results


# ──────────────────────────────────────────────────────────────────────
# 3. Visualisations
# ──────────────────────────────────────────────────────────────────────
def plot_behavior_pie(df, out_dir):
    """Pie chart: gcid behavior distribution."""
    counts = df["behavior"].value_counts()
    labels = []
    sizes = []
    colors = []
    color_map = {
        "Mostly Independent": INDEP_COLOR,
        "Mixed": MIX_COLOR,
        "Mostly Dependent": DEP_COLOR,
    }
    for beh in ["Mostly Independent", "Mixed", "Mostly Dependent"]:
        n = counts.get(beh, 0)
        labels.append(f"{beh} ({n})")
        sizes.append(n)
        colors.append(color_map[beh])

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        startangle=90, textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title(
        f"gcid Behavior Distribution - {SYSTEM} ({CLONE_TYPE})\n"
        f"n={len(df)} gcids | dep_ratio <= 10% = Indep, > 50% = Dep"
    )
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_gcid_behavior_pie.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_gcid_behavior_pie.png")


def plot_author_impact(df, out_dir):
    """Grouped bar chart: author -> gcid behavior counts."""
    auth = df.groupby("author").agg(
        n_gcids=("gcid", "count"),
        mostly_indep=("behavior", lambda x: (x == "Mostly Independent").sum()),
        mixed=("behavior", lambda x: (x == "Mixed").sum()),
        mostly_dep=("behavior", lambda x: (x == "Mostly Dependent").sum()),
    ).sort_values("n_gcids", ascending=False).head(8)

    if auth.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(auth))
    w = 0.25

    ax.bar(x - w, auth["mostly_indep"], w, label="Mostly Independent",
           color=INDEP_COLOR, edgecolor="white", linewidth=0.5)
    ax.bar(x, auth["mixed"], w, label="Mixed",
           color=MIX_COLOR, edgecolor="white", linewidth=0.5)
    ax.bar(x + w, auth["mostly_dep"], w, label="Mostly Dependent",
           color=DEP_COLOR, edgecolor="white", linewidth=0.5)

    # Annotate total gcids
    for i, (_, row) in enumerate(auth.iterrows()):
        ax.text(i, max(row["mostly_indep"], row["mixed"], row["mostly_dep"]) + 2,
                f"n={row['n_gcids']}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(auth.index, rotation=45, ha="right")
    ax.set_ylabel("Number of gcids")
    ax.set_title(
        f"Author Impact on gcid Independence - {SYSTEM}\n"
        f"How many gcids per author are independent, mixed, or dependent?"
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    sns.despine()

    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_gcid_author_impact.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_gcid_author_impact.png")


def plot_file_impact(df, test_results, out_dir):
    """Side-by-side comparison: gcids with same-file pairs vs only cross-file."""
    has_sf = df[df["same_file_frac"] > 0]
    no_sf = df[df["same_file_frac"] == 0]

    categories = ["Has Same-File Pairs", "Only Cross-File Pairs"]
    groups = [has_sf, no_sf]
    colors_bar = [DEP_COLOR, INDEP_COLOR]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: dep_ratio distribution
    ax = axes[0]
    for i, (lbl, grp, clr) in enumerate(zip(categories, groups, colors_bar)):
        ax.hist(grp["dep_ratio"], bins=20, alpha=0.6, label=f"{lbl} (n={len(grp)})",
                color=clr, edgecolor="white")
    ax.set_xlabel("Dependency Ratio")
    ax.set_ylabel("Number of gcids")
    ax.set_title("Distribution of gcid Dep Ratio\nby File Co-location")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    # Right: behavior breakdown
    ax = axes[1]
    for i, (lbl, grp, clr) in enumerate(zip(categories, groups, colors_bar)):
        mi = (grp["behavior"] == "Mostly Independent").sum()
        mx = (grp["behavior"] == "Mixed").sum()
        md = (grp["behavior"] == "Mostly Dependent").sum()
        bars = ax.bar(
            [i - 0.2, i, i + 0.2], [mi, mx, md], width=0.18,
            color=[INDEP_COLOR, MIX_COLOR, DEP_COLOR],
            edgecolor="white", linewidth=0.5,
        )
        for b, v in zip(bars, [mi, mx, md]):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                        ha="center", fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Number of gcids")

    r = test_results.get("pearson_samefile_dep_r", 0)
    p = test_results.get("pearson_samefile_dep_p", 1)
    ax.set_title(f"gcid Behavior by File Co-location\nPearson r={r:.3f}, p={p:.2e}")
    ax.grid(axis="y", alpha=0.2)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=INDEP_COLOR, label="Mostly Indep"),
        Patch(facecolor=MIX_COLOR, label="Mixed"),
        Patch(facecolor=DEP_COLOR, label="Mostly Dep"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.suptitle(f"File Location Impact on gcid Independence - {SYSTEM}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_gcid_file_impact.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_gcid_file_impact.png")


def plot_combined_heatmap(df, out_dir):
    """Heatmap: mean dep_ratio by author x has_same_file_pairs."""
    df_temp = df.copy()
    df_temp["has_sf"] = df_temp["same_file_frac"].apply(
        lambda x: "Same-file pairs" if x > 0 else "Cross-file only"
    )

    # Only authors with enough data
    top_authors = df_temp["author"].value_counts()
    top_authors = top_authors[top_authors >= 3].index.tolist()
    df_temp = df_temp[df_temp["author"].isin(top_authors)]

    pivot = df_temp.pivot_table(
        index="author", columns="has_sf", values="dep_ratio",
        aggfunc="mean"
    )
    # Reorder columns
    col_order = ["Cross-file only", "Same-file pairs"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    # Count matrix for annotation
    count_pivot = df_temp.pivot_table(
        index="author", columns="has_sf", values="dep_ratio",
        aggfunc="count"
    )
    count_pivot = count_pivot[[c for c in col_order if c in count_pivot.columns]]

    # Build annotation strings
    annot = pivot.astype(object)
    for r in annot.index:
        for c in annot.columns:
            val = pivot.loc[r, c] if pd.notna(pivot.loc[r, c]) else 0
            cnt = int(count_pivot.loc[r, c]) if pd.notna(count_pivot.loc[r, c]) else 0
            annot.loc[r, c] = f"{val:.1%}\n(n={cnt})"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot, annot=annot, fmt="", cmap="RdYlGn_r", ax=ax,
        linewidths=0.5, vmin=0, vmax=0.8,
        cbar_kws={"label": "Mean Dep Ratio"},
    )
    ax.set_xlabel("File Co-location")
    ax.set_ylabel("Primary Author")
    ax.set_title(
        f"Combined Effect: Author x File Location on gcid Dependency - {SYSTEM}\n"
        f"Values = mean dep_ratio, (n = number of gcids)"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{SYSTEM}_gcid_combined_heatmap.png"))
    plt.close(fig)
    print(f"  [OK] Saved {SYSTEM}_gcid_combined_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  gcid-Level Impact Analysis - {SYSTEM} ({CLONE_TYPE})")
    print("=" * 60 + "\n")

    # Build dataset
    print("[1/5] Building gcid-level dataset...")
    df, frag, pair = build_gcid_dataset()
    print(f"  Total gcids: {len(df)}")
    print(f"  Mean dep_ratio: {df['dep_ratio'].mean():.3f}")
    print(f"  Median dep_ratio: {df['dep_ratio'].median():.3f}")

    # Behavior distribution
    print("\n[2/5] gcid behavior distribution:")
    for beh in ["Mostly Independent", "Mixed", "Mostly Dependent"]:
        n = (df["behavior"] == beh).sum()
        pct = n / len(df) * 100
        print(f"  {beh:25s}: {n:4d} ({pct:.1f}%)")

    # Author impact
    print("\n  Author -> gcid behavior:")
    auth = df.groupby("author").agg(
        n_gcids=("gcid", "count"),
        mean_dep_ratio=("dep_ratio", "mean"),
        median_dep_ratio=("dep_ratio", "median"),
        mostly_indep=("behavior", lambda x: (x == "Mostly Independent").sum()),
        mixed=("behavior", lambda x: (x == "Mixed").sum()),
        mostly_dep=("behavior", lambda x: (x == "Mostly Dependent").sum()),
    ).sort_values("n_gcids", ascending=False)
    print(auth.to_string())

    # File impact
    print("\n  File co-location -> gcid behavior:")
    has_sf = df[df["same_file_frac"] > 0]
    no_sf = df[df["same_file_frac"] == 0]
    print(f"  WITH same-file pairs:  n={len(has_sf):3d}, mean dep_ratio={has_sf['dep_ratio'].mean():.3f}")
    print(f"  ONLY cross-file pairs: n={len(no_sf):3d}, mean dep_ratio={no_sf['dep_ratio'].mean():.3f}")

    # Combined
    print("\n  Combined (Author x File):")
    combo = df.groupby(["author", df["same_file_frac"] > 0]).agg(
        n=("gcid", "count"),
        mean_dr=("dep_ratio", "mean"),
    ).reset_index()
    combo.columns = ["author", "has_same_file_pairs", "n_gcids", "mean_dep_ratio"]
    combo["has_same_file_pairs"] = combo["has_same_file_pairs"].map({True: "Yes", False: "No"})
    print(combo.sort_values(["author", "has_same_file_pairs"]).to_string(index=False))

    # Save CSV
    print("\n[3/5] Saving gcid impact CSV...")
    csv_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_gcid_impact.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved {csv_path}")

    # Statistical tests
    print("\n[4/5] Running statistical tests...")
    test_results = run_gcid_tests(df)

    stats_lines = []
    stats_lines.append(f"gcid-Level Impact Analysis - {SYSTEM} ({CLONE_TYPE})")
    stats_lines.append("=" * 60)
    stats_lines.append("")
    stats_lines.append(f"Total gcids: {len(df)}")
    stats_lines.append(f"Mean dep_ratio: {df['dep_ratio'].mean():.4f}")
    stats_lines.append(f"Median dep_ratio: {df['dep_ratio'].median():.4f}")
    stats_lines.append("")

    stats_lines.append("1. File Co-location Impact (same_file_frac <-> dep_ratio)")
    stats_lines.append(f"   Pearson r: {test_results['pearson_samefile_dep_r']:.4f}")
    stats_lines.append(f"   p-value:   {test_results['pearson_samefile_dep_p']:.2e}")
    if "pointbiserial_hasfile_r" in test_results:
        stats_lines.append(f"   Point-biserial r (has_sf binary): {test_results['pointbiserial_hasfile_r']:.4f}")
        stats_lines.append(f"   Point-biserial p: {test_results['pointbiserial_hasfile_p']:.2e}")
    stats_lines.append("")

    if "mannwhitney_file_U" in test_results:
        stats_lines.append("2. Mann-Whitney U: dep_ratio (has same-file vs cross-file only)")
        stats_lines.append(f"   U: {test_results['mannwhitney_file_U']:.0f}")
        stats_lines.append(f"   p-value: {test_results['mannwhitney_file_p']:.2e}")
        stats_lines.append(f"   Has same-file (n={test_results['has_samefile_n']}): mean={test_results['has_samefile_mean_dr']:.4f}")
        stats_lines.append(f"   Cross-file only (n={test_results['no_samefile_n']}): mean={test_results['no_samefile_mean_dr']:.4f}")
        stats_lines.append("")

    if "anova_author_F" in test_results:
        stats_lines.append("3. ANOVA: dep_ratio ~ author")
        stats_lines.append(f"   F-statistic: {test_results['anova_author_F']:.2f}")
        stats_lines.append(f"   p-value: {test_results['anova_author_p']:.2e}")
        stats_lines.append("")

    stats_lines.append("CONCLUSION:")
    r_file = abs(test_results.get("pearson_samefile_dep_r", 0))
    f_auth = test_results.get("anova_author_F", 0)
    if r_file > 0.5:
        stats_lines.append(f"   File co-location is the DOMINANT driver (r={r_file:.3f}).")
        stats_lines.append("   A gcid is independent when its clone peers are in different files.")
        stats_lines.append("   A gcid is dependent when it shares a file with its clone peers.")
    if f_auth > 5:
        stats_lines.append(f"   Author identity is significant (F={f_auth:.1f}) but secondary.")
        stats_lines.append("   The same author produces both independent and dependent gcids")
        stats_lines.append("   depending on whether their gcids have same-file clone peers.")

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    stats_path = os.path.join(RESULTS_DIR, f"{SYSTEM}_gcid_impact_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\n  [OK] Saved {stats_path}")

    # Plots
    print("\n[5/5] Generating plots...")
    plot_behavior_pie(df, RESULTS_DIR)
    plot_author_impact(df, RESULTS_DIR)
    plot_file_impact(df, test_results, RESULTS_DIR)
    plot_combined_heatmap(df, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("  DONE - gcid impact analysis complete for", SYSTEM)
    print("=" * 60)


if __name__ == "__main__":
    main()
