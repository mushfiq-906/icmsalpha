#!/usr/bin/env python3
"""
run_all_analyses.py — Unified runner for all interpretability analyses.
=======================================================================
Executes all four standalone analyses in sequence and produces a
combined findings report with a ranked feature list.

Analyses:
  1. Author-Based Dependency Triggering  (author_analysis.py)
  2. File/Folder Structural Proximity    (file_folder_analysis.py)
  3. Clone Similarity & Class Size       (similarity_classsize_analysis.py)
  4. gcid-Level Impact                   (gcid_impact_analysis.py)

Output:
  - ml/results/combined_findings_{SYSTEM}.txt
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SYSTEM = "dnsjava"
CLONE_TYPE = "Type3_Block"
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "WorkFolder", SYSTEM, "Datasets", "CloneGenealogy"
)
REV_PAIR = os.path.join(BASE_DIR, f"{CLONE_TYPE}_rev_pair.csv")
REV_FRAGMENT = os.path.join(BASE_DIR, f"{CLONE_TYPE}_rev_fragment.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_analysis(name, module_name):
    """Run a single analysis module by importing and calling its main().

    Args:
        name: Human-readable name of the analysis.
        module_name: Python module name to import.

    Returns:
        bool: True if successful, False if failed.
    """
    print(f"\n{'#' * 70}")
    print(f"  RUNNING: {name}")
    print(f"{'#' * 70}\n")
    t0 = time.time()
    try:
        mod = __import__(module_name)
        mod.main()
        elapsed = time.time() - t0
        print(f"\n  >> {name} completed in {elapsed:.1f}s\n")
        return True
    except Exception as e:
        print(f"\n  >> {name} FAILED: {e}\n")
        return False


def compute_combined_rankings():
    """Compute point-biserial correlations for all features vs dep_label.

    Loads rev_pair.csv and rev_fragment.csv to derive all analysed features,
    then ranks them by absolute point-biserial correlation with dep_label.

    Returns:
        tuple: (rankings_df, findings_dict) where rankings_df is a sorted
               DataFrame of features and findings_dict contains key conclusions.
    """
    pair = pd.read_csv(REV_PAIR)
    frag = pd.read_csv(REV_FRAGMENT)

    pair["dep_label"] = (
        (pair["changeType1"] != "U") & (pair["changeType2"] != "U")
    ).astype(int)

    # Derive author features
    author_map = frag[["revision", "gcid", "author"]].drop_duplicates()
    merged = pair.merge(
        author_map.rename(columns={"gcid": "gcid1", "author": "author1"}),
        on=["revision", "gcid1"], how="left",
    )
    merged = merged.merge(
        author_map.rename(columns={"gcid": "gcid2", "author": "author2"}),
        on=["revision", "gcid2"], how="left",
    )

    # Derived features
    both_changed = (merged["changeType1"] != "U") & (merged["changeType2"] != "U")
    has_both = both_changed & merged["author1"].notna() & merged["author2"].notna()
    merged["derived_same_author"] = np.nan
    merged.loc[has_both, "derived_same_author"] = (
        merged.loc[has_both, "author1"] == merged.loc[has_both, "author2"]
    ).astype(int)

    # All numeric features to rank
    features = {
        "sameFile": merged["sameFile"],
        "depth": merged["depth"],
        "similarity": merged["similarity"],
        "classSize": merged["classSize"],
        "sameAuthor (raw)": merged["sameAuthor"],
        "coChangeCount": merged["coChangeCount"],
        "weightedCouplingStrength": merged["weightedCouplingStrength"],
        "couplingTrend": merged["couplingTrend"],
    }

    dep = merged["dep_label"]
    rankings = []
    for feat_name, feat_vals in features.items():
        valid = feat_vals.notna() & dep.notna()
        if valid.sum() < 10:
            continue
        try:
            r, p = stats.pointbiserialr(dep[valid], feat_vals[valid])
            rankings.append({
                "feature": feat_name,
                "point_biserial_r": round(r, 4),
                "abs_r": round(abs(r), 4),
                "p_value": p,
                "significant": "Yes" if p < 0.05 else "No",
                "direction": "positive" if r > 0 else "negative",
            })
        except Exception:
            pass

    rank_df = pd.DataFrame(rankings).sort_values("abs_r", ascending=False).reset_index(drop=True)
    rank_df["rank"] = range(1, len(rank_df) + 1)

    # Key findings
    findings = {}

    # Strongest author feature
    auth_feats = rank_df[rank_df["feature"].str.contains("author|Author", case=False)]
    if len(auth_feats) > 0:
        findings["strongest_author"] = auth_feats.iloc[0]["feature"]
        findings["strongest_author_r"] = auth_feats.iloc[0]["point_biserial_r"]

    # sameFile vs depth
    sf_r = rank_df.loc[rank_df["feature"] == "sameFile", "abs_r"].values
    dp_r = rank_df.loc[rank_df["feature"] == "depth", "abs_r"].values
    if len(sf_r) > 0 and len(dp_r) > 0:
        if sf_r[0] > dp_r[0]:
            findings["spatial_winner"] = "sameFile"
            findings["spatial_winner_r"] = float(sf_r[0])
        else:
            findings["spatial_winner"] = "depth"
            findings["spatial_winner_r"] = float(dp_r[0])

    # similarity vs classSize
    sim_r = rank_df.loc[rank_df["feature"] == "similarity", "abs_r"].values
    cls_r = rank_df.loc[rank_df["feature"] == "classSize", "abs_r"].values
    if len(sim_r) > 0 and len(cls_r) > 0:
        if sim_r[0] > cls_r[0]:
            findings["structural_winner"] = "similarity"
            findings["structural_winner_r"] = float(sim_r[0])
        else:
            findings["structural_winner"] = "classSize"
            findings["structural_winner_r"] = float(cls_r[0])

    return rank_df, findings


def write_combined_report(rank_df, findings, results_status):
    """Write the combined findings report.

    Args:
        rank_df: DataFrame of ranked features.
        findings: dict of key conclusions.
        results_status: dict mapping analysis names to pass/fail.
    """
    lines = []
    lines.append(f"COMBINED FINDINGS REPORT - {SYSTEM} ({CLONE_TYPE})")
    lines.append("=" * 70)
    lines.append("")

    lines.append("Analysis Execution Status:")
    for name, ok in results_status.items():
        status = "PASS" if ok else "FAIL"
        lines.append(f"  [{status}] {name}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)
    lines.append("")

    # 1. Author
    if "strongest_author" in findings:
        lines.append(f"1. Strongest author feature triggering dependency:")
        lines.append(f"   {findings['strongest_author']} (r = {findings['strongest_author_r']:.4f})")
    else:
        lines.append("1. Author features: insufficient variation for ranking")
    lines.append("")

    # 2. Spatial
    if "spatial_winner" in findings:
        lines.append(f"2. Stronger spatial predictor of co-change:")
        lines.append(f"   {findings['spatial_winner']} (|r| = {findings['spatial_winner_r']:.4f})")
        lines.append(f"   >> {'sameFile' if findings['spatial_winner']=='sameFile' else 'depth'} "
                      f"is the stronger spatial predictor")
    lines.append("")

    # 3. Structural
    if "structural_winner" in findings:
        lines.append(f"3. Stronger structural predictor of co-change:")
        lines.append(f"   {findings['structural_winner']} (|r| = {findings['structural_winner_r']:.4f})")
        lines.append(f"   >> {'similarity' if findings['structural_winner']=='similarity' else 'classSize'} "
                      f"is the stronger structural predictor")
    lines.append("")

    # 4. Ranked list
    lines.append("-" * 70)
    lines.append("RANKED FEATURE LIST (by |point-biserial r| with dep_label)")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"{'Rank':<6}{'Feature':<30}{'r':<12}{'|r|':<10}{'p-value':<14}{'Direction':<12}{'Sig?'}")
    lines.append("-" * 94)
    for _, row in rank_df.iterrows():
        lines.append(
            f"{row['rank']:<6}{row['feature']:<30}{row['point_biserial_r']:<12.4f}"
            f"{row['abs_r']:<10.4f}{row['p_value']:<14.2e}{row['direction']:<12}{row['significant']}"
        )
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    report_path = os.path.join(RESULTS_DIR, f"combined_findings_{SYSTEM}.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  [OK] Saved {report_path}")


def main():
    t_total = time.time()
    print("=" * 70)
    print(f"  UNIFIED ANALYSIS RUNNER - {SYSTEM} ({CLONE_TYPE})")
    print("=" * 70)

    # Run all analyses
    status = {}
    status["1. Author-Based Dependency"] = run_analysis(
        "Author-Based Dependency Triggering", "author_analysis")
    status["2. File/Folder Proximity"] = run_analysis(
        "File/Folder Structural Proximity", "file_folder_analysis")
    status["3. Similarity & ClassSize"] = run_analysis(
        "Clone Similarity & Class Size", "similarity_classsize_analysis")
    status["4. gcid-Level Impact"] = run_analysis(
        "gcid-Level Impact Analysis", "gcid_impact_analysis")

    # Combined rankings
    print(f"\n{'#' * 70}")
    print("  COMPUTING COMBINED RANKINGS")
    print(f"{'#' * 70}\n")

    rank_df, findings = compute_combined_rankings()

    # Write report
    write_combined_report(rank_df, findings, status)

    elapsed = time.time() - t_total
    print(f"\n  TOTAL TIME: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
