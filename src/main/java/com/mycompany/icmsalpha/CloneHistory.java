package com.mycompany.icmsalpha;

import java.util.ArrayList;
import java.util.List;

class CloneHistory {
    int globalCloneId;
    int startRev; // first revision where the clone appeared
    int endRev; // last revision where the clone existed
    List<Integer> changeRevs = new ArrayList<>();
    List<Integer> unchangedRevs = new ArrayList<>();
    int addedInRev; // revision where it first appeared
    int deletedInRev; // revision where it disappeared

    // Evolution Forecasting Fields (Dr. Mondal's methodology)
    double stabilityIndex; // ratio of unchanged to total revisions (0.0 - 1.0)
    double changeProneness; // frequency of changes (0.0 - 1.0)
    double independentEvolutionScore; // likelihood of independent evolution (0.0 - 1.0)
    String latePropagationStatus; // "NONE", "DETECTED", "RESOLVED"
    List<Integer> latePropagationRevs = new ArrayList<>(); // revisions where late propagation occurred
    int coChangeCount; // number of revisions where clone changed with peers
    int independentChangeCount; // number of revisions where clone changed alone
    // Process metrics (computed post-lifecycle by computeProcessAndOwnershipMetrics)
    int noc;            // Number of Changes = changeRevs.size()
    int fileAge;        // Lifetime in revisions = endRev - addedInRev
    int churn;          // True lines changed in clone region (from ChangeInfo diff hunks)
    // Ownership metrics (computed from revAuthorMap + changeRevs)
    int distinctAuthors;             // # unique authors who committed changes to this clone
    double majorAuthorProportion;    // TCO: fraction of commits by top author (0.0 - 1.0)
    int minorAuthorCount;            // # authors contributing < 5% of total commits
}
