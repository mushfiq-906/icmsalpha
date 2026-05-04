# icmsalpha

This is the repository for my master's thesis work. I am working on **forecasting independent evolution possibilities of code clones**.

## Overview

Code clones (duplicated code fragments) often evolve together (co-change) but sometimes diverge — one copy is modified while the other is not, leading to inconsistent maintenance and latent bugs. This project builds a machine-learning pipeline that **predicts whether a clone pair will independently evolve** (diverge) in the next revision, using historical clone genealogy data.

## Research Question

> Given the evolution history of a clone pair up to revision *n*, can we predict whether the pair will diverge (independently evolve) at revision *n+1*?

## Approach

### 1. Clone Tracking (Java side)
The `icmsalpha` Java module extracts clone genealogies from version-control history of subject systems. It produces per-revision records of clone pairs annotated with structural and behavioral metrics.

### 2. Feature Engineering
For each clone pair at each revision, we compute:

| Feature | Meaning |
|---|---|
| **IR** | Inconsistent Revisions — count of past divergent revisions |
| **CS** | Co-change Score — frequency of joint modification |
| **SPCP** | Sibling Pattern Co-change Probability |
| **IR_decay**, **CS_decay**, **SPCP_decay** | Time-decayed versions (5 half-lives) emphasizing recent history |

### 3. Forecasting Models
Four supervised classifiers plus a soft-voting ensemble:

- **LightGBM**
- **XGBoost**
- **CatBoost**
- **Random Forest**
- **Ensemble** — averaged probabilities of the three boosters

### 4. Evaluation: Walk-Forward (Prequential)
A single 70/30 split is misleading on this data because the positive (independent-evolution) class is rare and concentrated in late revisions. We use **walk-forward evaluation** instead:

```
For each event index i in [MIN_TRAIN, n):
    train  = df.iloc[:i]            (all events strictly before i)
    test   = df.iloc[i:i+1]         (the single event at i)
```

- The model is **refit every `REFIT_EVERY` events** (default: 5).
- A **threshold** is tuned on a sliding `CALIB_WINDOW` (default: 50) of recent events to adapt to local label-distribution drift.
- Metrics aggregate over the full prediction stream.

This better matches deployment: at revision *n*, predict using only history `1..n-1`.

## Subject Systems & Clone Types

| Subject | Clone Types Studied |
|---|---|
| **Ctags** | Type-1, Type-2, Type-3 (Block-level) |
| **JMol** | Type-3 (Block-level) |

Datasets are organized under:

```
WorkFolder/<Subject>/Datasets/CloneGenealogy/
    Type<N>_Block_evolution_dataset.csv     ← features + target
    Type<N>_Block_forecast_dataset.csv      ← held-out forecast set
```

## Repository Structure

```
icmsalpha/
├── src/main/java/com/mycompany/icmsalpha/   # Java clone-tracking pipeline
├── WorkFolder/
│   ├── Ctags/Datasets/CloneGenealogy/        # Ctags datasets
│   └── Jmol/Datasets/CloneGenealogy/         # JMol datasets
├── ml/
│   ├── train_test.py                         # static 70/30 baseline + helpers
│   ├── walk_forward.py                       # prequential evaluation (primary)
│   └── results/                              # metrics, plots, trained models
├── catboost_info/                            # CatBoost training logs
└── README.md
```

## Running the ML Pipeline

```bash
# Walk-forward evaluation on Ctags Type-3
python ml/walk_forward.py --csv WorkFolder/Ctags/Datasets/CloneGenealogy/Type3_Block_evolution_dataset.csv

# JMol Type-3 with the decay-weighted target
python ml/walk_forward.py --csv WorkFolder/Jmol/Datasets/CloneGenealogy/Type3_Block_evolution_dataset.csv --target will_diverge_decay
```

Outputs (suffixed with the subject/clone-type prefix):
- `ml/results/walk_forward_metrics_<PREFIX>.csv`
- `ml/results/walk_forward_predictions_<PREFIX>.csv`
- `ml/results/mcc_over_time_<PREFIX>.png`

## Current Results

### Walk-forward evaluation on JMol Type-3 (13,239 predictions)

| Model | MCC | AUC-ROC | PR-AUC | Balanced Acc | G-mean | F1 | Time (s) |
|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.9553** | 0.9971 | 0.9987 | 0.9754 | 0.9753 | 0.9880 | 2055.7 |
| Ensemble | 0.9529 | 0.9980 | 0.9993 | 0.9735 | 0.9733 | 0.9874 | 8220.1 |
| XGBoost | 0.9509 | 0.9973 | 0.9990 | 0.9731 | 0.9729 | 0.9869 | 2143.1 |
| CatBoost | 0.9506 | 0.9974 | 0.9991 | 0.9725 | 0.9723 | 0.9868 | 4021.3 |
| RandomForest | 0.8153 | 0.9958 | 0.9985 | 0.8718 | 0.8632 | 0.9529 | 3075.9 |

Imbalance-aware metrics (MCC, Balanced Acc, G-mean) are the primary indicators because of class skew. **LightGBM is the top single model**; the ensemble is comparable but ~4× slower.

## Configuration Knobs (`ml/walk_forward.py`)

| Constant | Default | Effect |
|---|---|---|
| `MIN_TRAIN` | 100 | Warm-up: events before first prediction |
| `REFIT_EVERY` | 5 | Refit cadence (lower = more responsive, slower) |
| `CALIB_WINDOW` | 50 | Recent events used for threshold tuning |
| `ROLLING_WINDOW` | 50 | Window for the MCC-over-time plot |

Each booster currently uses `n_estimators=250` to balance accuracy and runtime.

## Tooling

- **Java 8+** for `icmsalpha` clone tracking
- **Python 3.10+** with `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `matplotlib`

## Author

Master's thesis project — code-clone evolution forecasting.
