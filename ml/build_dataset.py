"""
build_dataset.py
────────────────────────────────────────────────────────────────────────────
Converts the Java-produced evolution_dataset CSV into a per-event
forecasting dataset.

Prediction setup
────────────────
At each change event at revision n for a clone pair:

    FEATURES = state of the pair at the PREVIOUS event (history 1..n-1)
    LABEL    = what type of change occurred AT revision n?
                 0 → co-change   (both clones changed together)
                 1 → independent (one or both clones changed alone)

The cumulative counts already in the CSV make the label trivially derivable:
    delta_co = co_change_count[n] − co_change_count[n-1]
    if delta_co == 1 → co-change  → will_diverge = 0
    if delta_co == 0 → independent→ will_diverge = 1

Features are obtained by shifting each pair's rows by 1 (pandas .shift(1)),
so the feature row for event n is the state snapshot at event n-1.

First event per pair is dropped (no prior state to use as features).

Usage:
    python build_dataset.py
    python build_dataset.py --csv path/to/evolution_dataset.csv

Output (written next to the input CSV):
    <name>_forecast_dataset.csv
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────

DEFAULT_CSV = (
    r"WorkFolder\Ctags\Datasets\CloneGenealogy"
    r"\Type3_Block_evolution_dataset.csv"
)

PAIR_COLS = ["gcid1", "gcid2"]

# Columns that are NOT features — excluded before shifting
NON_FEATURE_COLS = {
    "Revision",                    # kept for temporal split; not a feature
    "gcid1", "gcid2", "classid",  # identifiers
    "wies",                        # = 1 − WCS, redundant
    "will_independently_evolve",   # old leaky label — replaced here
    "will_diverge_decay",          # Option C label — not a feature for Option B
    "weighted_coupling_strength",  # subsumed by ir_decay_* and cs_decay_*
                                   # (WCS uses λ=0.03 ≈ h=23, nearly identical
                                   # to ir_decay_h20; keeping it causes all
                                   # decay variants to be pruned as duplicates)
}


# ── LABEL: what type of event occurred at this revision? ─────────────────────

def compute_event_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (pair, revision) row, determine whether the event that
    triggered this row was a co-change or an independent change.

    co_change_count is cumulative.  If it increased since the previous
    row → both clones changed → co-change.
    If it stayed the same → at least one clone changed alone → independent.
    """
    df = df.sort_values(PAIR_COLS + ["Revision"]).reset_index(drop=True)

    prev_co = (
        df.groupby(PAIR_COLS)["co_change_count"]
          .shift(1)
          .fillna(0)           # first event per pair: assume 0 prior co-changes
    )
    delta_co = df["co_change_count"] - prev_co

    # delta_co == 1 → co-change (label 0)
    # delta_co == 0 → independent (label 1)
    df["will_diverge"] = (delta_co == 0).astype(int)

    print(f"Label balance  : {df['will_diverge'].value_counts().to_dict()}")
    return df


# ── FEATURES: shift to previous-row state ────────────────────────────────────

def shift_features_to_previous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace every feature column with its value from the PREVIOUS event
    for the same pair.

    Row at revision n  ->  features become the state at revision n-1.
    First event per pair has no previous state -> dropped.
    """
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS
                 and c not in {"will_diverge", "will_independently_evolve",
                               "will_diverge_decay"}]

    df[feat_cols] = df.groupby(PAIR_COLS)[feat_cols].shift(1)

    before = len(df)
    df = df.dropna(subset=[feat_cols[0]]).reset_index(drop=True)
    print(f"Dropped {before - len(df):,} first-event rows (no prior state)")

    return df


# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features — all computed from already-shifted (history 1..n-1)
    column values.  No lookahead.
    Columns that don't exist in the CSV are silently skipped.
    """
    def safe(s):
        return s.replace(0, np.nan)

    def has(*cols):
        return all(c in df.columns for c in cols)

    # Change frequency rate
    if has("noc1", "file_age1"):
        df["change_freq1"] = (df["noc1"] / safe(df["file_age1"])).fillna(0)
    if has("noc2", "file_age2"):
        df["change_freq2"] = (df["noc2"] / safe(df["file_age2"])).fillna(0)

    # Co-change confidence: P(co-change | clone changed)
    if has("co_change_count", "noc1"):
        df["co_change_conf1"] = (df["co_change_count"] / safe(df["noc1"])).fillna(0)
    if has("co_change_count", "noc2"):
        df["co_change_conf2"] = (df["co_change_count"] / safe(df["noc2"])).fillna(0)

    # Co-change ratio: fraction of all pair events that were co-changes
    if has("co_change_count", "gcid1_independent_change_count",
           "gcid2_independent_change_count"):
        total_events = (
            df["co_change_count"]
            + df["gcid1_independent_change_count"]
            + df["gcid2_independent_change_count"]
        )
        df["co_change_ratio"] = (df["co_change_count"] / safe(total_events)).fillna(0)

    # Total independent events so far
    if has("gcid1_independent_change_count", "gcid2_independent_change_count"):
        df["total_ind_events"] = (
            df["gcid1_independent_change_count"]
            + df["gcid2_independent_change_count"]
        )

    # Churn intensity per edit
    if has("churn1", "noc1"):
        df["churn_per_change1"] = (df["churn1"] / safe(df["noc1"])).fillna(0)
    if has("churn2", "noc2"):
        df["churn_per_change2"] = (df["churn2"] / safe(df["noc2"])).fillna(0)

    # Asymmetry between the two clones
    if has("nlines1", "nlines2"):
        df["delta_nlines"] = (df["nlines1"] - df["nlines2"]).abs()
    if has("churn1", "churn2"):
        df["delta_churn"] = (df["churn1"] - df["churn2"]).abs()
    if has("noc1", "noc2"):
        df["delta_noc"] = (df["noc1"] - df["noc2"]).abs()
    if has("file_age1", "file_age2"):
        df["delta_file_age"] = (df["file_age1"] - df["file_age2"]).abs()
    if has("distinct_authors1", "distinct_authors2"):
        df["delta_distinct_authors"] = (
            df["distinct_authors1"] - df["distinct_authors2"]
        ).abs()
    if has("major_author_prop1", "major_author_prop2"):
        df["delta_major_author"] = (
            df["major_author_prop1"] - df["major_author_prop2"]
        ).abs()
    # ── Decay-weighted derived features ────────────────────────────────────

    # IR decay spread: difference between shortest and longest half-life
    # Large spread → IR estimate is sensitive to recency → unstable pair
    if has("ir_decay_h10", "ir_decay_h75"):
        df["ir_decay_spread"] = df["ir_decay_h10"] - df["ir_decay_h75"]

    # CS-IR divergence: when CS says coupled but IR says independent (or vice versa)
    if has("cs_decay_h20", "ir_decay_h20"):
        df["cs_ir_divergence_h20"] = df["cs_decay_h20"] - (1.0 - df["ir_decay_h20"])

    # Composite decay signal: average of IR_decay across half-lives
    ir_cols = [c for c in df.columns if c.startswith("ir_decay_h")]
    if len(ir_cols) > 0:
        df["ir_decay_mean"] = df[ir_cols].mean(axis=1)

    # Composite SPCP consistency: std of SPCP_decay across half-lives
    spcp_cols = [c for c in df.columns if c.startswith("spcp_decay_h")]
    if len(spcp_cols) > 1:
        df["spcp_decay_std"] = df[spcp_cols].std(axis=1)

    return df


# ── ENCODE ───────────────────────────────────────────────────────────────────

def encode(df: pd.DataFrame) -> pd.DataFrame:
    df["clone_type"] = df["clone_type"].astype("category").cat.codes
    return df


# ── PRUNE: drop constants and near-duplicate columns ─────────────────────────

PROTECTED = {"Revision", "gcid1", "gcid2", "classid", "will_diverge",
             "will_diverge_decay"}

# Always keep these even if constant in the current CSV.  They vary across
# clone types (e.g. similarity is constant only in Type-1/Type-2 because
# NiCad enforces a fixed threshold; in Type-3 it varies) and we want the
# column present so the same model schema works for every Type.
KEEP_EVEN_IF_CONSTANT = {"similarity", "same_author"}


def prune_features(df: pd.DataFrame, corr_threshold: float = 0.98) -> pd.DataFrame:
    """
    Drop columns that carry no information:
      Rule A - zero variance (nunique <= 1).
      Rule B - one of any pair with |Pearson corr| >= corr_threshold.
    Never touches PROTECTED columns or KEEP_EVEN_IF_CONSTANT columns.
    """
    feats = [c for c in df.columns if c not in PROTECTED]

    # Rule A
    constants = [c for c in feats
                 if df[c].nunique(dropna=False) <= 1
                 and c not in KEEP_EVEN_IF_CONSTANT]
    if constants:
        print(f"\nDropping {len(constants)} constant column(s): {constants}")
        df = df.drop(columns=constants)
        feats = [c for c in feats if c not in constants]
    kept_constant = [c for c in feats
                     if c in KEEP_EVEN_IF_CONSTANT
                     and df[c].nunique(dropna=False) <= 1]
    if kept_constant:
        print(f"Kept constant (cross-Type schema): {kept_constant}")

    # Rule B
    numeric = df[feats].select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        cols = list(corr.columns)
        corr_arr = corr.to_numpy()
        to_drop = set()
        print(f"\nNear-duplicate pairs (|corr| >= {corr_threshold}):")
        for i, a in enumerate(cols):
            if a in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                b = cols[j]
                if b in to_drop:
                    continue
                c_ab = float(corr_arr[i, j])
                if c_ab >= corr_threshold:
                    print(f"  drop {b!r}  (corr {c_ab:.3f} with {a!r})")
                    to_drop.add(b)
        if to_drop:
            df = df.drop(columns=list(to_drop))
            print(f"Dropped {len(to_drop)} near-duplicate column(s)")
        else:
            print("  (none)")

    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────

def build(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded  {len(df):,} rows × {df.shape[1]} cols")
    print(f"Revision range : {df['Revision'].min()} – {df['Revision'].max()}")

    # Drop old leaky label and redundant wies
    drop = [c for c in ["wies", "will_independently_evolve"] if c in df.columns]
    df = df.drop(columns=drop)

    # Step 1: label each event (before shifting, so we see the delta)
    df = compute_event_label(df)

    # Step 2: shift features to previous-row state
    df = shift_features_to_previous(df)

    # Step 3: derive engineered features from the (now shifted) values
    df = engineer_features(df)

    # Step 4: encode categoricals
    df = encode(df)

    # Step 5: prune constants and near-duplicate columns
    df = prune_features(df)

    # Sort globally by Revision so the output is in temporal order
    df = df.sort_values("Revision").reset_index(drop=True)

    pos = df["will_diverge"].mean()
    rev_min = df["Revision"].min()
    rev_max = df["Revision"].max()

    # Row-based 70/30 split — this matches what train_test.py actually uses
    # (SPLIT_BY_ROWS=True). Revision-range splits are misleading on this
    # dataset because change events cluster late, so 70% of the revision
    # range catches only ~25% of the rows.
    cut_idx = int(len(df) * 0.70)
    cut_rev_at_idx = int(df.sort_values("Revision").iloc[cut_idx]["Revision"])
    n_train = cut_idx
    n_test  = len(df) - cut_idx

    print(f"\nFinal dataset  : {len(df):,} rows x {df.shape[1]} cols")
    print(f"Revision range : {rev_min} - {rev_max}")
    print(f"Label balance  : {1-pos:.1%} co-change / {pos:.1%} independent")
    print(f"\nTemporal split preview (70/30 on row count, matches train_test.py):")
    print(f"  TRAIN : first {n_train:,} rows  (revision <= {cut_rev_at_idx})")
    print(f"  TEST  : last  {n_test:,} rows   (revision >  {cut_rev_at_idx})")

    out = Path(csv_path).parent / (
        Path(csv_path).stem.replace("evolution_dataset", "forecast_dataset") + ".csv"
    )
    df.to_csv(out, index=False)
    print(f"\nSaved  -> {out.name}")
    return df, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()
    build(args.csv)
