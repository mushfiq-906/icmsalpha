# Integrating Decay-Weighted Metrics into Clone Evolution Pipeline

## Goal

Integrate three exponential time-decay weighted metrics — **IR_decay**, **CS_decay**, and **SPCP_decay** — computed at 5 candidate half-lives (h ∈ {10, 20, 30, 50, 75}), into both the Java backend and Python ML pipeline. Primary setup: **Option B** (decay in features, full-history labels). Sensitivity analysis: **Option C** (decay in both).

## Decisions Resolved

| Question | Decision | Rationale |
|----------|----------|-----------|
| Multiple λ strategy | **Option A**: Export 5 sets of decay features (15 new cols) | Let model/Optuna naturally select optimal half-life via feature importance |
| CS_decay denominator | **Keep both WCS and CS_decay** as separate features | WCS (Jaccard/union) is symmetric and penalizes size differences; CS_decay (max denominator) is asymmetric and captures containment. Both provide complementary signals — [research shows](https://towardsdatascience.com) Jaccard and overlap-style metrics capture different aspects of co-change behavior. Mondal's work uses Jaccard-style union, but the advisor's max formulation captures "what fraction of the more-active clone's changes are co-changes" — useful for asymmetric pairs |
| SPCP_decay | **Re-run fragment similarity dynamically** at each revision | More accurate than parsing static CSV; computes per-revision SPCP status |
| Dormant pair filter | **Already implemented** | Java: only creates records when `changed1 \|\| changed2` (line 990); Python: drops first event per pair via shift (line 104), so pairs with ≤1 event are excluded |

## Current Labeling Mechanism (for context)

The pipeline has two labeling layers:

1. **Java-side** (`will_independently_evolve`): Based on `WIES = 1 − WCS`. If WCS < 0.5, labeled independent. This IS decay-weighted (λ=0.03).
2. **Python-side** (`will_diverge`): Based on `delta_co_change_count`. If the event at revision n had no co-change → labeled independent (1). This is **unweighted per-event ground truth**.

The Python label **drops and replaces** the Java label. For Option B, `will_diverge` stays as the primary label. For Option C sensitivity analysis, a new `will_diverge_decay` label uses the composite decay-weighted decision hierarchy.

## Proposed Changes

---

### Component 1: Java Backend — CloneGenealogyAnalysis.java

#### [MODIFY] [CloneGenealogyAnalysis.java](file:///d:/Thesis/icmsalpha/src/main/java/com/mycompany/icmsalpha/CloneGenealogyAnalysis.java)

##### 1a. Add half-life constants and λ computation

Replace the single `DECAY_LAMBDA = 0.03` with an array of 5 half-lives:

```java
private static final int[] HALF_LIVES = {10, 20, 30, 50, 75};

private static double lambdaFromHalfLife(int h) {
    return Math.log(2.0) / h;
}
```

##### 1b. Add IR_decay method

New method computing the decay-weighted Independence Ratio:

```java
/**
 * IR_decay = Σ w(indep_changes) / (Σ w(co_changes) + Σ w(indep_changes) + ε)
 */
private double computeDecayWeightedIR(CloneHistory h1, CloneHistory h2,
        int minRev, int maxRev, double lambda) {
    double coWeight = 0.0, indepWeight = 0.0;
    // ... iterate changeRevs, classify co vs independent, apply e^(-λ*age)
}
```

##### 1c. Add CS_decay method (advisor's max-denominator formulation)

New method — distinct from existing WCS (which uses union denominator):

```java
/**
 * CS_decay = Σ w(co_changes) / max(Σ w(gc1_all_changes), Σ w(gc2_all_changes)) + ε
 */
private double computeDecayWeightedCS(CloneHistory h1, CloneHistory h2,
        int minRev, int maxRev, double lambda) {
    // Numerator: weighted co-change sum
    // Denominator: max of weighted individual change sums
}
```

##### 1d. Extend EvolutionRecord with 15 decay-feature fields + 1 label

```java
// 5 half-lives × 3 metrics = 15 new fields
double[] irDecay  = new double[5]; // ir_decay_h10..h75
double[] csDecay  = new double[5]; // cs_decay_h10..h75
double[] spcpDecay = new double[5]; // spcp_decay_h10..h75

// Option C label (using default h=20, index 1)
int willDivergeDecay; // 1 if composite hierarchy says independent
```

##### 1e. Compute metrics in exportEvolutionDataset() loop

Inside the `if (changed1 || changed2)` block (after line 1050), compute all 15 metrics:

```java
for (int hi = 0; hi < HALF_LIVES.length; hi++) {
    double lambda = lambdaFromHalfLife(HALF_LIVES[hi]);
    rec.irDecay[hi] = computeDecayWeightedIR(h1, h2, pairStartRev, rev, lambda);
    rec.csDecay[hi] = computeDecayWeightedCS(h1, h2, pairStartRev, rev, lambda);
    rec.spcpDecay[hi] = computeDecayWeightedSPCP(h1, h2, spcpRevisions, rev, lambda);
}

// Option C label: composite decision using h=20 (index 1)
boolean dependent = rec.csDecay[1] >= 0.5
    && rec.spcpDecay[1] >= 0.5
    && rec.irDecay[1] < 0.5;
rec.willDivergeDecay = dependent ? 0 : 1;
```

##### 1f. Extend CSV header and output

Add 16 new columns to the CSV header and format string:
```
ir_decay_h10,ir_decay_h20,ir_decay_h30,ir_decay_h50,ir_decay_h75,
cs_decay_h10,cs_decay_h20,cs_decay_h30,cs_decay_h50,cs_decay_h75,
spcp_decay_h10,spcp_decay_h20,spcp_decay_h30,spcp_decay_h50,spcp_decay_h75,
will_diverge_decay
```

---

### Component 2: Java Backend — SPCP_decay with Dynamic Fragment Similarity

#### [MODIFY] [CloneGenealogyAnalysis.java](file:///d:/Thesis/icmsalpha/src/main/java/com/mycompany/icmsalpha/CloneGenealogyAnalysis.java)

For SPCP_decay, we need to determine at each co-change revision whether the change was similarity-preserving. This requires:

1. **Fragment fetching** — reuse the approach from `SPCPAnalysis.java`'s `FileBasedFragmentFetcher` (reads source files at specific revisions and extracts clone fragments by line range)

2. **LCS similarity** — reuse `SPCPAnalysis.java`'s `lcsSimilarity()` method

3. **Per-revision SPCP classification** — for each co-change revision, check if similarity is preserved after the change

```java
/**
 * Dynamically compute SPCP status for each co-change revision of a pair.
 * Returns the set of revision numbers where the co-change was similarity-preserving.
 */
private Set<Integer> computeSPCPRevisions(int gcid1, int gcid2,
        CloneHistory h1, CloneHistory h2, int pairStartRev, int maxRev) {
    Set<Integer> coChangeRevs = new TreeSet<>(h1.changeRevs);
    coChangeRevs.retainAll(h2.changeRevs);

    Set<Integer> spcpRevs = new TreeSet<>();
    for (int rev : coChangeRevs) {
        if (rev > maxRev) continue;
        int nextRev = findNextRevWithBothClones(gcid1, gcid2, rev);
        if (nextRev == -1) continue;

        String before1 = getFragmentAt(gcid1, rev);
        String before2 = getFragmentAt(gcid2, rev);
        String after1  = getFragmentAt(gcid1, nextRev);
        String after2  = getFragmentAt(gcid2, nextRev);

        if (before1 == null || before2 == null || after1 == null || after2 == null) continue;

        double simAfter = lcsSimilarity(after1, after2);
        if (simAfter >= SIM_THRESHOLD) {
            spcpRevs.add(rev);
        }
    }
    return spcpRevs;
}

/**
 * SPCP_decay = Σ w(SPCP-verified revisions) / (Σ w(all co-change revisions) + ε)
 */
private double computeDecayWeightedSPCP(Set<Integer> spcpRevs,
        Set<Integer> allCoChangeRevs, int maxRev, double lambda) {
    double spcpWeight = 0.0, totalWeight = 0.0;
    for (int rev : allCoChangeRevs) {
        double w = Math.exp(-lambda * (maxRev - rev));
        totalWeight += w;
        if (spcpRevs.contains(rev)) spcpWeight += w;
    }
    return totalWeight < 1e-9 ? 0.0 : spcpWeight / totalWeight;
}
```

> [!IMPORTANT]
> This requires loading clone instance data (file path, start/end lines) per revision into `CloneGenealogyAnalysis`, similar to how `SPCPAnalysis` builds `cloneInstancesByRevision`. We need to add a `Map<Integer, Map<Integer, Clones>> cloneInstancesByRevision` field and populate it during the data loading phase of `exportEvolutionDataset()`.

> [!WARNING]
> **Performance concern**: Dynamic fragment comparison at every co-change revision for every pair could be expensive (file I/O + LCS computation). For a project with 100 revisions and ~50 pairs, this is manageable (~500 fragment comparisons). For larger datasets, we should add a fragment cache. The existing `SPCPAnalysis` already handles this scale successfully.

---

### Component 3: Python ML Pipeline

#### [MODIFY] [build_dataset.py](file:///d:/Thesis/icmsalpha/ml/build_dataset.py)

1. Add all 16 new columns to `NON_FEATURE_COLS` that shouldn't be features:
   - `will_diverge_decay` (Option C label — not a feature)

2. Keep the 15 decay metric columns as features (they ARE valid features for Option B)

3. Add derived features in `engineer_features()`:
   ```python
   # IR decay trend: difference between smallest and largest half-life IR
   if has("ir_decay_h10", "ir_decay_h75"):
       df["ir_decay_spread"] = df["ir_decay_h10"] - df["ir_decay_h75"]
   ```

4. The existing `prune_features()` with `corr_threshold=0.98` will naturally thin highly correlated half-life variants

#### [MODIFY] [train_test.py](file:///d:/Thesis/icmsalpha/ml/train_test.py)

1. Add `--target` CLI argument (default: `will_diverge`)
2. Add `will_diverge_decay` to `NON_FEATURES`
3. For Option C sensitivity: `python train_test.py --target will_diverge_decay`

#### [MODIFY] [walk_forward.py](file:///d:/Thesis/icmsalpha/ml/walk_forward.py)

1. Same `--target` CLI argument
2. Inherit `will_diverge_decay` in `NON_FEATURES`

---

## File Change Summary

| File | Changes | New Lines (est.) |
|------|---------|-----------------|
| `CloneGenealogyAnalysis.java` | 4 new methods, extend EvolutionRecord, extend CSV output, add fragment loading | ~250 |
| `build_dataset.py` | Add to NON_FEATURE_COLS, add derived features | ~20 |
| `train_test.py` | Add `--target` arg, add to NON_FEATURES | ~10 |
| `walk_forward.py` | Add `--target` arg | ~5 |

## Verification Plan

### Automated Tests

1. **Numerical verification** (advisor's example): Co-changes at revisions 15, 30, 45; independent at 60, 75, 90; r_current=100, λ=0.03 (h≈23):
   - Expected IR_decay ≈ 0.79 (vs unweighted 0.50)
   - Build a manual test case in the Java code to verify

2. **CSV column verification**: After Java export:
   ```bash
   head -1 evolution_dataset.csv | tr ',' '\n' | grep -c decay
   # Should output 16 (15 features + 1 label)
   ```

3. **ML pipeline smoke test**:
   ```bash
   python ml/build_dataset.py --csv <path>
   python ml/train_test.py --csv <forecast_csv>                        # Option B
   python ml/train_test.py --csv <forecast_csv> --target will_diverge_decay  # Option C
   ```

4. **SHAP analysis**: After training, inspect which half-life features rank highest — this reveals the optimal decay rate as a thesis finding

### Manual Verification
- Compare MCC/AUC between Option B and Option C
- Report optimal half-life as empirical finding
