# Theoretical Guide: Code Clone Evolution Research

> **Thesis Title:** "Forecasting Independent Evolution Possibilities of Code Clones"

---

## Table of Contents
1. [Code Clone Fundamentals](#1-code-clone-fundamentals)
2. [Clone Types (Type 1, 2, 3)](#2-clone-types)
3. [Clone Evolution](#3-clone-evolution)
4. [SPCP - Similarity Preserving Change Patterns](#4-spcp---similarity-preserving-change-patterns)
5. [Clone Genealogy](#5-clone-genealogy)
6. [Clone Tracking Across Revisions](#6-clone-tracking-across-revisions)
7. [Independent Evolution](#7-independent-evolution-forecasting-target)
8. [Research Methodology](#8-research-methodology)
9. [Tools: NiCad, ctags, diff](#9-tools-reference)
10. [ML Forecasting Approach](#10-ml-forecasting-approach)
11. [Key References](#11-key-references)

---

## 1. Code Clone Fundamentals

### Definition
A **code clone** is a code fragment that has similar or identical fragments elsewhere in the codebase. Clones are typically detected at:
- **Function/Method level** (most common for evolution studies)
- **Block level** (loops, conditionals)
- **File level** (entire file duplication)

### Why Clones Exist

| Reason | Description | Example |
|--------|-------------|---------|
| **Copy-Paste Programming** | Fastest development approach | Copying handler code with minor changes |
| **Forking** | Creating variants from existing code | Platform-specific implementations |
| **Templated Code** | Patterns required by frameworks | Getter/setter methods |
| **Language Limitations** | No abstraction mechanism available | Macro expansions in C |
| **Independent Solutions** | Same problem solved similarly | Sorting algorithms |

### Clone Relationship Terminology

```
┌─────────────────────────────────────────────────────────────┐
│                    CLONE VOCABULARY                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Clone Fragment: A single piece of duplicated code          │
│  ─────────────                                              │
│  Clone Pair: Two similar code fragments                     │
│  ──────────                                                 │
│  Clone Class: A set of similar fragments (≥2 fragments)     │
│  ───────────                                                │
│  Clone Group: Same as Clone Class                           │
│  ───────────                                                │
│                                                             │
│  Example Clone Class with 3 fragments:                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │Fragment 1│──│Fragment 2│──│Fragment 3│                   │
│  │ (File A) │  │ (File B) │  │ (File C) │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Clone Types

### Type-1: Exact Clones
Identical code except for **whitespace, layout, and comments**.

```java
// === ORIGINAL (File: UserService.java, Line 45-50) ===
public void saveUser(User user) {
    validate(user);
    repository.save(user);
    log.info("User saved");
}

// === TYPE-1 CLONE (File: AdminService.java, Line 78-82) ===
public void saveUser(User user) {
  validate(user);repository.save(user);
  log.info("User saved"); // Same logic, different formatting
}
```

**Detection:** NiCad with threshold = 0%, after pretty-printing normalization.

---

### Type-2: Renamed/Parameterized Clones
Identical structure with **different identifiers, literals, types**.

```java
// === ORIGINAL ===
public void saveUser(User user) {
    validate(user);
    userRepository.save(user);
    logger.info("User saved: " + user.getId());
}

// === TYPE-2 CLONE ===
public void saveProduct(Product product) {
    validate(product);
    productRepository.save(product);
    logger.info("Product saved: " + product.getId());
}
```

**Detection:** NiCad with blind renaming or consistent renaming.

| Renaming Mode | Description |
|---------------|-------------|
| **Blind Renaming** | All identifiers replaced with same placeholder |
| **Consistent Renaming** | Same identifiers get same placeholder |

---

### Type-3: Near-Miss/Gapped Clones
Similar code with **added, deleted, or modified statements**.

```java
// === ORIGINAL ===
public void saveUser(User user) {
    validate(user);
    userRepository.save(user);
    logger.info("User saved");
}

// === TYPE-3 CLONE (30% different) ===
public void saveUserWithAudit(User user) {
    if (user == null) {                    // ADDED
        throw new IllegalArgumentException();
    }
    validate(user);
    auditService.logAction("SAVE_USER");   // ADDED
    userRepository.save(user);
    notifyAdmin(user);                     // ADDED
    logger.info("User saved with audit");  // MODIFIED
}
```

**Detection:** NiCad with threshold (e.g., 0.30 = 30% difference allowed).

---

### Type Comparison Matrix

| Aspect | Type-1 | Type-2 | Type-3 |
|--------|--------|--------|--------|
| **Textual Similarity** | Near 100% | Near 100% after renaming | 70-99% |
| **Structural Similarity** | Identical | Identical | Similar |
| **Identifiers** | Same | Different | May differ |
| **Statements** | Same | Same | May differ |
| **NiCad Threshold** | 0% | 0% | 0-30% |
| **Evolution Risk** | Low | Medium | High |

---

## 3. Clone Evolution

### Definition
**Clone evolution** describes how code clones change over time across multiple revisions/versions of a software system.

### Evolution States

```
┌────────────────────────────────────────────────────────────────┐
│                    CLONE EVOLUTION STATES                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Rev(n)            Rev(n+1)          Description               │
│  ──────            ────────          ───────────               │
│                                                                │
│  ┌────┐            ┌────┐                                      │
│  │ C  │ ────────→ │ C  │           STABLE (No change)         │
│  └────┘            └────┘                                      │
│                                                                │
│  ┌────┐            ┌────┐                                      │
│  │ C  │ ────────→ │ C' │           CHANGED (Modified)         │
│  └────┘            └────┘                                      │
│                                                                │
│  ┌────┐            ┌────┐                                      │
│  │ C  │ ────────→ │    │           DEAD (Deleted)             │
│  └────┘            └────┘                                      │
│                                                                │
│  ┌────┐            ┌────┐ ┌────┐                               │
│  │ C  │ ────────→ │C1  │ │C2  │    SPLIT (Divided)            │
│  └────┘            └────┘ └────┘                               │
│                                                                │
│  ┌────┐ ┌────┐    ┌────┐                                       │
│  │C1  │ │C2  │ →  │ C  │           MERGED (Combined)          │
│  └────┘ └────┘    └────┘                                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Clone Class Evolution Patterns

For a clone class (group of fragments), evolution can be:

| Pattern | Description | All Fragments |
|---------|-------------|---------------|
| **All Same** | All fragments unchanged | Stable |
| **All Changed Consistently** | All fragments changed identically | Maintained together |
| **Some Changed** | Some fragments changed, others stable | **Inconsistent** |
| **All Changed Differently** | Each fragment changed uniquely | **Independent evolution** |
| **Some Deleted** | Some fragments removed | Partial death |
| **All Deleted** | Clone class disappeared | Complete death |

---

## 4. SPCP - Similarity Preserving Change Patterns

### Definition
**SPCP (Similarity Preserving Change Pattern)** categorizes how clone pairs/classes maintain or lose their similarity when changed.

> [!IMPORTANT]
> SPCP is **central to your thesis** - it determines whether clones evolve **consistently** (similarity preserved) or **independently** (similarity lost).

### SPCP Categories

```
┌─────────────────────────────────────────────────────────────────┐
│              SIMILARITY PRESERVING CHANGE PATTERNS               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pattern        │ Clone 1  │ Clone 2  │ Similarity │ Category   │
│  ───────────────┼──────────┼──────────┼────────────┼────────────│
│                 │          │          │            │            │
│  NO CHANGE      │ Same     │ Same     │ Preserved  │ Consistent │
│                 │          │          │            │            │
│  SAME CHANGE    │ Changed  │ Same     │ Preserved  │ Consistent │
│                 │ to X     │ Change   │            │            │
│                 │          │ to X     │            │            │
│                 │          │          │            │            │
│  PROPORTIONAL   │ +5 lines │ +5 lines │ Partially  │ Mostly     │
│  CHANGE         │ (similar)│ (similar)│ Preserved  │ Consistent │
│                 │          │          │            │            │
│  INCONSISTENT   │ Changed  │ Different│ LOST       │ DIVERGING  │
│  CHANGE         │ to X     │ Change Y │            │            │
│                 │          │          │            │            │
│  UNILATERAL     │ Changed  │ No       │ LOST       │ DIVERGING  │
│  CHANGE         │          │ Change   │            │            │
│                 │          │          │            │            │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed SPCP Example

**Initial State (Rev 1):**
```java
// Clone Fragment A (UserService.java:45)
public void process(Data data) {
    validate(data);
    save(data);
    notify(data);
}

// Clone Fragment B (OrderService.java:78)  
public void process(Data data) {
    validate(data);
    save(data);
    notify(data);
}
```
*Similarity: 100% (Type-1 clone pair)*

---

**Scenario 1: SAME CHANGE (Similarity Preserved)**
```java
// Rev 2: Clone Fragment A
public void process(Data data) {
    if (data == null) return;  // Added same line
    validate(data);
    save(data);
    notify(data);
}

// Rev 2: Clone Fragment B
public void process(Data data) {
    if (data == null) return;  // Added same line
    validate(data);
    save(data);
    notify(data);
}
```
*Result: Similarity preserved ✓ → Consistent evolution*

---

**Scenario 2: UNILATERAL CHANGE (Similarity Lost)**
```java
// Rev 2: Clone Fragment A (CHANGED)
public void process(Data data) {
    if (data == null) return;  // Added
    validate(data);
    auditLog(data);            // Added
    save(data);
    notify(data);
    metrics.record("process"); // Added
}

// Rev 2: Clone Fragment B (UNCHANGED)
public void process(Data data) {
    validate(data);
    save(data);
    notify(data);
}
```
*Result: Similarity lost ✗ → Independent evolution (YOUR TARGET)*

---

**Scenario 3: INCONSISTENT CHANGE (Both Changed Differently)**
```java
// Rev 2: Clone Fragment A
public void process(Data data) {
    if (data == null) return;  // Added validation
    validate(data);
    save(data);
    notify(data);
}

// Rev 2: Clone Fragment B
public void process(Data data) {
    validate(data);
    save(data);
    sendEmail(data);           // Added different feature
    notify(data);
    logAction("processed");    // Added different feature
}
```
*Result: Similarity lost ✗ → Independent evolution (YOUR TARGET)*

---

### SPCP Measurement

**Similarity Calculation:**
```
                    2 × LCS(TokensA, TokensB)
Similarity(A, B) = ──────────────────────────────
                     |TokensA| + |TokensB|

Where LCS = Longest Common Subsequence
```

**SPCP Classification Rules:**

| Condition | SPCP Category |
|-----------|---------------|
| Sim(Rev n) ≈ Sim(Rev n+1) | PRESERVED |
| Sim(Rev n) > Sim(Rev n+1) by small amount | DEGRADED |
| Sim(Rev n) >> Sim(Rev n+1) | LOST (Independent) |
| Clone dropped below threshold | SEPARATED |

---

## 5. Clone Genealogy

### Definition
A **clone genealogy** is a directed acyclic graph (DAG) representing the historical evolution of clone classes across all revisions.

### Genealogy Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLONE GENEALOGY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Revision 1      Revision 2      Revision 3      Revision 4    │
│  ──────────      ──────────      ──────────      ──────────    │
│                                                                 │
│  Clone Class 1   Clone Class 1   Clone Class 1                 │
│  ┌───┬───┐       ┌───┬───┐       ┌───┬───┐       (Deleted)     │
│  │C1a│C1b│  ──→  │C1a│C1b│  ──→  │C1a│C1b│                     │
│  └───┴───┘       └───┴───┘       └───┴───┘                     │
│                                                                 │
│  Clone Class 2   Clone Class 2   Clone Class 2   Clone Class 2 │
│  ┌───┬───┐       ┌───┬───┐       ┌───┬───┬───┐   ┌───┬───┬───┐ │
│  │C2a│C2b│  ──→  │C2a│C2b│  ──→  │C2a│C2b│C2c│→ │C2a│C2b│C2c│ │
│  └───┴───┘       └───┴───┘       └───┴───┴───┘   └───┴───┴───┘ │
│                                  (C2c added)                    │
│                                                                 │
│  Clone Class 3                   Clone Class 3                  │
│  ┌───┬───┬───┐                   ┌───┐   ┌───┐                  │
│  │C3a│C3b│C3c│  ──→  ──→  ──→    │C3a│   │C3b│  (C3a,C3b split) │
│  └───┴───┴───┘                   └───┘   └───┘   C3c deleted    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Genealogy Events

| Event | Description | Research Implication |
|-------|-------------|---------------------|
| **BIRTH** | New clone class detected | Track from this point |
| **STABLE** | Clone class unchanged | Low maintenance risk |
| **GROW** | New fragment added to class | Clone spreading |
| **SHRINK** | Fragment removed from class | Partial refactoring |
| **CHANGE_CONSISTENT** | All fragments changed same way | Good maintenance |
| **CHANGE_INCONSISTENT** | Fragments changed differently | **BUG RISK** |
| **SPLIT** | Class divides into separate classes | **Independent evolution** |
| **MERGE** | Classes combine | Consolidation |
| **DEATH** | Class no longer exists | Refactored/deleted |

---

## 6. Clone Tracking Across Revisions

### The Tracking Problem
Given clones detected in Rev(n) and Rev(n+1), how do we determine if a clone in Rev(n+1) is the "same" clone from Rev(n)?

### Tracking Algorithm Steps

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLONE TRACKING ALGORITHM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: LOCATION MATCHING                                      │
│  ─────────────────────────                                      │
│  • Same file path? (exact or with path tolerance)               │
│  • Similar line position? (within tolerance, e.g., ±50 lines)   │
│                                                                 │
│  Step 2: CONTENT SIMILARITY                                     │
│  ────────────────────────                                       │
│  • Calculate token-based similarity                             │
│  • Check if similarity > threshold (e.g., 70%)                  │
│                                                                 │
│  Step 3: CONTEXT MATCHING                                       │
│  ───────────────────────                                        │
│  • Same containing method/function?                             │
│  • Same containing class?                                       │
│                                                                 │
│  Step 4: ASSIGN GLOBAL ID                                       │
│  ───────────────────────                                        │
│  • If match found → propagate existing globalCloneId            │
│  • If no match → assign new globalCloneId                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tracking Challenges

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **Code Movement** | File renamed or moved | Track by content, not just path |
| **Line Shift** | Lines added/deleted above | Use line tolerance window |
| **Content Change** | Clone modified | Use similarity threshold |
| **Split/Merge** | Clone divided or combined | Track fragments individually |

### Your Implementation (from `CloneDetection.java`)
Your `getRestGlobalClone4` method implements this tracking:
```java
// Location check
boolean sameFile = compareFilePaths(prevClone.filePath, currClone.filePath);
// Position check with tolerance
boolean similarPosition = isLinePositionSimilar(prevClone, currClone);
// Content similarity
boolean similarContent = isItSameCode(prevClone, currClone);
```

---

## 7. Independent Evolution (Forecasting Target)

### Definition
**Independent evolution** occurs when clone fragments that were once similar **diverge** in their changes over time.

> [!CAUTION]
> Independent evolution is problematic because:
> - Bug fixes may miss some fragments
> - Features become inconsistent
> - Technical debt accumulates

### Detecting Independent Evolution

```
┌─────────────────────────────────────────────────────────────────┐
│             INDEPENDENT EVOLUTION DETECTION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For Clone Class with fragments {F1, F2, F3} at Rev(n):        │
│                                                                 │
│  1. Get fragments at Rev(n+1): {F1', F2', F3'}                  │
│                                                                 │
│  2. Calculate change deltas:                                    │
│     Δ1 = diff(F1, F1')                                          │
│     Δ2 = diff(F2, F2')                                          │
│     Δ3 = diff(F3, F3')                                          │
│                                                                 │
│  3. Compare deltas:                                             │
│                                                                 │
│     IF Δ1 ≈ Δ2 ≈ Δ3:                                            │
│        → CONSISTENT evolution                                   │
│                                                                 │
│     IF Δ1 ≠ Δ2 OR Δ2 ≠ Δ3:                                      │
│        → INDEPENDENT evolution (YOUR TARGET!)                   │
│                                                                 │
│  4. Record SPCP pattern                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Features That May Predict Independent Evolution

| Feature Category | Features | Hypothesis |
|-----------------|----------|------------|
| **Location** | Same file?, Same package?, Distance | Clones in different files evolve independently more often |
| **Clone Properties** | Type, Size, Age | Type-3 clones may diverge faster |
| **Historical** | Past change frequency, Past SPCP | Prior independent changes predict future |
| **Authorship** | Same author?, #Authors | Multiple authors → more divergence |
| **Coupling** | Shared dependencies?, Same callers? | Low coupling → independent evolution |
| **Method Context** | Method complexity, Method size | Complex methods change more |

---

## 8. Research Methodology

### Phase 1: Subject System Selection

**Criteria for SourceForge Projects:**
- Language: C, C#, or Java
- Minimum 50 revisions (ideally 100+)
- Active development period of 2+ years
- Codebase size: 10K+ LOC
- Multiple contributors (5+)

**Recommended Projects:**
- Games (active development, frequent changes)
- Libraries (stable API, internal refactoring)
- Applications (feature evolution)

---

### Phase 2: Data Collection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Subject System with N Revisions                         │
│  ─────────────────────────────────────                          │
│                                                                 │
│  For each revision r (1 to N):                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. Extract Methods using ctags                            │ │
│  │     └── Output: methods_rev{r}.txt                         │ │
│  │                                                            │ │
│  │  2. Run NiCad Clone Detection                              │ │
│  │     ├── Type-1 clones → type1_rev{r}.xml                   │ │
│  │     ├── Type-2 clones → type2_rev{r}.xml                   │ │
│  │     └── Type-3 clones → type3_rev{r}.xml                   │ │
│  │                                                            │ │
│  │  3. Parse NiCad Output                                     │ │
│  │     └── Store in database with global clone IDs            │ │
│  │                                                            │ │
│  │  4. Track Clone Evolution                                  │ │
│  │     └── Map clones to previous revision                    │ │
│  │                                                            │ │
│  │  5. Calculate SPCP                                         │ │
│  │     └── Determine if similarity preserved                  │ │
│  │                                                            │ │
│  │  6. Extract Features                                       │ │
│  │     └── Clone-level, method-level, history features        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Output: Feature dataset with target variable                   │
│  ─────────────────────────────────────────                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Phase 3: Feature Dataset Schema

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET SCHEMA                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PRIMARY KEY                                                    │
│  ───────────                                                    │
│  global_clone_id    INT          Unique clone identifier        │
│  revision           INT          Source revision number         │
│                                                                 │
│  CLONE FEATURES                                                 │
│  ──────────────                                                 │
│  clone_type         ENUM(1,2,3)  Type classification            │
│  clone_size_loc     INT          Lines of code                  │
│  clone_age          INT          # revisions since birth        │
│  clone_class_size   INT          # fragments in class           │
│                                                                 │
│  LOCATION FEATURES                                              │
│  ─────────────────                                              │
│  same_file_ratio    FLOAT        % fragments in same file       │
│  same_package_ratio FLOAT        % fragments in same package    │
│  avg_distance       FLOAT        Avg line distance              │
│                                                                 │
│  HISTORY FEATURES                                               │
│  ────────────────                                               │
│  change_frequency   FLOAT        Changes per revision           │
│  consistency_ratio  FLOAT        Consistent/Total changes       │
│  last_spcp_pattern  ENUM         Last SPCP category             │
│                                                                 │
│  METHOD FEATURES                                                │
│  ───────────────                                                │
│  method_complexity  FLOAT        McCabe complexity              │
│  method_size        INT          Method LOC                     │
│  method_params      INT          # of parameters                │
│                                                                 │
│  TARGET VARIABLE                                                │
│  ───────────────                                                │
│  will_evolve_independently  BOOLEAN  1 if diverges in next N    │
│                                      revisions, 0 otherwise     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Phase 4: Train/Test Split Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                 TEMPORAL TRAIN/TEST SPLIT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ⚠️ NEVER use random split (data leakage!)                      │
│                                                                 │
│  CORRECT: Temporal Split                                        │
│  ───────────────────────                                        │
│                                                                 │
│  Revision 1                     Revision 70      Revision 100   │
│  │═══════════════════════════════│───────────────│              │
│  │                               │               │              │
│  │       TRAINING SET            │  TEST SET     │              │
│  │         (70%)                 │    (30%)      │              │
│  │                               │               │              │
│  │  Learn patterns from          │ Predict on    │              │
│  │  historical data              │ future data   │              │
│  │                               │               │              │
│  └───────────────────────────────┴───────────────┘              │
│                                                                 │
│  Cross-System Validation:                                       │
│  • Train on System A, Test on System B                          │
│  • Tests generalizability                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Tools Reference

### NiCad Clone Detector

**What it does:**
- Text-based clone detection using TXL
- Pretty-printing normalizes formatting
- Configurable thresholds for Type-3 detection

**Configuration (nicad6config/):**
```
granularity = functions    # Detect at function level
threshold = 0.30           # 30% difference allowed for Type-3
minsize = 5                # Minimum 5 lines
rename = blind             # Blind renaming for Type-2
```

**Output Format (XML):**
```xml
<clones>
  <classinfo nclones="2" similarity="100">
    <source file="UserService.java" startline="45" endline="52"/>
    <source file="OrderService.java" startline="78" endline="85"/>
  </classinfo>
</clones>
```

---

### ctags (Universal Ctags)

**What it does:**
- Extracts method/function signatures
- Provides start line numbers
- Supports C, C++, C#, Java, and more

**Usage:**
```bash
ctags --fields=+neS --output-format=u-ctags -o tags.txt *.java
```

**Output:**
```
save    UserService.java    45    signature:(User user)    kind:method
```

---

### Diff Tools

**What it does:**
- Compares files between revisions
- Identifies added/deleted/modified lines
- Calculates change metrics

**Types:**
- `diff` (Unix standard)
- `git diff` (for version control)
- Custom token-based diff (for accuracy)

---

## 10. ML Forecasting Approach

### Problem Formulation

```
┌─────────────────────────────────────────────────────────────────┐
│                  ML PROBLEM FORMULATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Feature vector X for a clone at revision r              │
│  Output: Probability that clone will evolve independently       │
│          in next N revisions                                    │
│                                                                 │
│  P(independent_evolution | X) = f(X)                            │
│                                                                 │
│  Where X = [clone_type, clone_size, same_file_ratio, ...]       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Recommended Algorithms

| Algorithm | Pros | Cons |
|-----------|------|------|
| **Random Forest** | Interpretable, handles imbalance | May overfit |
| **XGBoost/LightGBM** | High accuracy, fast | Less interpretable |
| **Logistic Regression** | Simple, interpretable | May underfit |
| **Neural Network** | Captures complex patterns | Needs more data |

### Evaluation Metrics

| Metric | Formula | Use When |
|--------|---------|----------|
| **Accuracy** | (TP+TN)/(Total) | Balanced classes |
| **Precision** | TP/(TP+FP) | Cost of false positive high |
| **Recall** | TP/(TP+FN) | Cost of missing true positive high |
| **F1-Score** | 2×(P×R)/(P+R) | Imbalanced classes |
| **AUC-ROC** | Area under ROC | Overall performance |

### Feature Importance Analysis

After training, analyze feature importance:
```
┌─────────────────────────────────────────────────────────────────┐
│                 EXPECTED FEATURE IMPORTANCE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Feature                    │  Expected Importance              │
│  ────────────────────────── │  ────────────────                  │
│  same_file_ratio            │  High (location matters)          │
│  past_consistency_ratio     │  High (history repeats)           │
│  clone_type                 │  Medium (Type-3 diverges more)    │
│  clone_age                  │  Medium (older = more stable)     │
│  author_count               │  Medium (more authors = diverge)  │
│  method_complexity          │  Low-Medium                       │
│                                                                 │
│  This tells you WHY clones evolve independently!                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Key References

### Clone Detection
1. Roy, C.K., Cordy, J.R. (2008). "NiCad: Accurate Detection of Near-miss Intentional Clones Using Flexible Pretty-Printing and Code Normalization"

### Clone Evolution
2. Krinke, J. (2007). "A Study of Consistent and Inconsistent Changes to Code Clones"
3. Göde, N., Koschke, R. (2009). "Incremental Clone Detection"
4. Göde, N., Koschke, R. (2011). "Studying Clone Evolution Using Incremental Clone Detection"

### Clone Genealogy
5. Kim, M., et al. (2005). "An Empirical Study of Code Clone Genealogies"
6. Duala-Ekoko, E., Robillard, M.P. (2007). "Tracking Code Clones in Evolving Software"

### Clone Management
7. Juergens, E., et al. (2009). "Do Code Clones Matter?"
8. Harder, J., Göde, N. (2011). "Efficiently Handling Clone Data"

---

> [!TIP]
> **Next Steps:**
> 1. Select 3-5 subject systems from SourceForge
> 2. Configure NiCad for Type-1, Type-2, Type-3 detection
> 3. Build data collection pipeline
> 4. Extract features and create ML dataset
> 5. Train and evaluate forecasting model
