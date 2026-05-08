# icmsalpha

This is the repository for my master's thesis work on **forecasting dependent and independent evolution of code clone fragments**.

## Overview

Code clones (duplicated code fragments) often evolve together (co-change) but sometimes diverge — one copy is modified while the other is not, leading to inconsistent maintenance and latent bugs. This project builds a machine-learning pipeline that **predicts whether a changed clone fragment will change dependently** (co-changing with at least one clone-class peer) **or independently** (changing in isolation) at its next modification event, using historical clone genealogy data.

## Research Question

> Given the evolution history of clone fragments up to revision *R−1*, can we predict whether each fragment that changes at revision *R* will change **dependently** or **independently**?

## Methodology

### 1. Clone Genealogy Extraction (Java)

The `icmsalpha` Java module extracts clone genealogies from version-control history using NiCad clone detection. For each revision, every clone fragment is tracked by a persistent **global clone identifier (gcid)** with a change type: **M** (modified), **U** (unchanged), or **A** (added).

Two revision-wise datasets are generated:

- **`rev_fragment.csv`** — Fragment-level features (one row per modified fragment per revision)
- **`rev_pair.csv`** — Pair-level coupling features (one row per active pair per revision where at least one side changed)

### 2. Feature Engineering (52 dimensions)

At each validation revision *R*, a 52-dimensional feature vector is built for each changed fragment using **only data from revisions strictly before *R*** (no look-ahead bias):

**Fragment-level features (12):** nlines, totalChanges, totalUnchanged, stabilityIndex, changeProneness, lifespan, activePairs, meanWCS, maxWCS, minWCS, stronglyCoupledPairs, decoupledPairs

**Side-aware process metrics (4):** churn, distinctAuthors, majorAuthorProp, minorAuthorCount

**Aggregated pair features (13):** pair_count_alive, max/sum co-change counts, max/sum independent counts, coupling_trend, last_co_change_age, last_solo_change_age, similarity, sameFile, depth

**Enriched pair features (8):** wcsRecent, soloEventsAfterLastCoChange, isSpcp, classSize, sameAuthor

**Decay-weighted metrics (15):** IR_decay, CS_decay, SPCP_decay at 5 half-lives (h ∈ {10, 20, 30, 50, 75})

### 3. Labeling

- **Dependent (label = 1):** At least one clone-class peer also changed at revision *R*
- **Independent (label = 0):** No peer changed — the fragment changed alone

### 4. Classification Models

Four supervised classifiers evaluated independently:

| Model | Key Parameters | Class Balancing |
|-------|---------------|-----------------|
| **LightGBM** | 300 rounds, depth=6, lr=0.05 | `is_unbalance=True` |
| **XGBoost** | 300 rounds, depth=6, lr=0.05 | `scale_pos_weight` (auto) |
| **CatBoost** | 300 iterations, depth=6, lr=0.05 | `auto_class_weights="Balanced"` |
| **Random Forest** | 300 trees, depth=10, min_leaf=5 | `class_weight="balanced"` |

### 5. Walk-Forward Evaluation

A **sliding-window test-then-train** protocol that prevents data leakage:

```
For each validation revision Rᵢ:
  1. Build feature vectors using only data < Rᵢ
  2. Predict dependent/independent for each changed fragment
  3. Validate against actual labels at Rᵢ
  4. Add verified (x, y) to training pool
  5. Retrain model (REFIT_EVERY = 1)
  6. Re-calibrate threshold via MCC optimization
```

### 6. Evaluation Metrics

MCC (primary), AUC-ROC, PR-AUC, Balanced Accuracy, G-mean, F1, Sensitivity, Specificity

## Current Results — Ctags Type-3 Block

**1,179 predictions** across 191 change revisions (485 unique fragments):

| Model | MCC | AUC-ROC | PR-AUC | Bal.Acc | G-mean | F1 | Sens | Spec |
|-------|-----|---------|--------|---------|--------|-----|------|------|
| **LightGBM** | **0.578** | 0.825 | 0.896 | **0.790** | **0.787** | **0.867** | 0.864 | 0.716 |
| CatBoost | 0.562 | 0.834 | 0.901 | 0.783 | 0.780 | 0.861 | 0.855 | 0.711 |
| RandomForest | 0.561 | **0.861** | 0.910 | 0.784 | 0.781 | 0.859 | 0.849 | **0.719** |
| XGBoost | 0.497 | 0.858 | **0.921** | 0.755 | 0.753 | 0.833 | 0.811 | 0.700 |

Label distribution: ~69% dependent, ~31% independent.

## Repository Structure

```
icmsalpha/
├── src/main/java/com/mycompany/icmsalpha/   # Java clone-tracking & dataset export
├── WorkFolder/
│   └── Ctags/Datasets/CloneGenealogy/
│       ├── Type3_Block_rev_fragment.csv      # Fragment-level revision data
│       └── Type3_Block_rev_pair.csv          # Pair-level revision data (50 columns)
├── ml/
│   ├── predict_standing.py                   # Walk-forward fragment-level forecasting
│   ├── plots.py                              # Analysis graph generation
│   └── results/                              # Metrics, plots, feature importance
└── README.md
```

## Running the Pipeline

```bash
# Fragment-level walk-forward prediction on Ctags Type-3
python ml/predict_standing.py \
  --frag WorkFolder/Ctags/Datasets/CloneGenealogy/Type3_Block_rev_fragment.csv \
  --pair WorkFolder/Ctags/Datasets/CloneGenealogy/Type3_Block_rev_pair.csv
```

### Output Files

| File | Description |
|------|-------------|
| `standing_*_global_metrics.csv` | Final metrics per model |
| `standing_*_verification_log.csv` | Per-event prediction log |
| `standing_*_per_revision.csv` | Per-revision rolling metrics |
| `standing_*_rolling_mcc.png` | MCC over revisions |
| `standing_*_confusion_matrix.png` | Confusion matrices |
| `standing_*_roc_curve.png` | ROC curves |
| `standing_*_pr_curve.png` | Precision-Recall curves |
| `standing_*_feature_importance_*.png/csv` | Feature importance per model |
| `standing_*_model_comparison.png` | Model comparison bar chart |
| `standing_*_label_distribution.png` | Label distribution pie chart |

## Configuration (`predict_standing.py`)

| Constant | Default | Effect |
|----------|---------|--------|
| `WARMUP_REVS` | 5 | Revisions before first prediction |
| `MIN_TRAIN` | 30 | Minimum training samples before fitting |
| `REFIT_EVERY` | 1 | Retrain after every revision |
| `CALIB_WINDOW` | 100 | Recent events for threshold calibration |

## Tooling

- **Java 8+** for clone genealogy extraction and dataset export
- **Python 3.10+** with `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `matplotlib`

## Author

Master's thesis project — code clone fragment change forecasting.
