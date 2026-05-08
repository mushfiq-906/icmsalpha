package com.mycompany.icmsalpha;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Integrated CloneGenealogyAnalysis:
 * - discovers clones appearing in later revisions
 * - builds revision -> author mapping from commit_logs.txt (infers missing
 * revisions)
 * - stores filepath per globalCloneId
 * - outputs CSV with authors and filepath
 */
public class CloneGenealogyAnalysis {

    private int choice;
    private String granularity;
    private String gPath, cChoice;
    private int startRev, endRev;
    AccessPoint ap = new AccessPoint();
    CommonParameters cp = new CommonParameters();

    /**
     * Exponential decay rate for time-decay weighted coupling strength.
     * Higher lambda = faster decay of old events. Tune between 0.01 and 0.05.
     */
    private static final double DECAY_LAMBDA = 0.03;
    private static final int    WCS_RECENT_WINDOW = 10; // revision window for short-term coupling

    /**
     * Candidate half-lives (in revisions) for multi-λ decay feature export.
     * λ = ln(2) / h.  The model/Optuna selects the best via feature importance.
     * h=20 → events 60 revisions old carry ~12.5% weight (3 half-lives).
     */
    private static final int[] HALF_LIVES = {10, 20, 30, 50, 75};

    /** SPCP similarity threshold — loaded from config, same as SPCPAnalysis */
    private double spcpSimThreshold = 0.70; // default, overridden in constructor if config exists

    private static double lambdaFromHalfLife(int h) {
        return Math.log(2.0) / h;
    }

    // Path to commit logs (searches common relative locations)
    private final List<String> commitSearchPaths = Arrays.asList(
            "./commit_logs.txt",
            "commit_logs.txt",
            "../commit_logs.txt");

    public CloneGenealogyAnalysis(int choice, String granularity, int startRev, int endRev) {
        this.startRev = startRev;
        this.endRev = endRev;

        if (granularity.equals("f")) {
            this.granularity = "Function";
        } else if (granularity.equals("b")) {
            this.granularity = "Block";
        }

        if (choice == 1) {
            this.cChoice = "Type1";
        } else if (choice == 2) {
            this.cChoice = "Type2";
        } else if (choice == 3) {
            this.cChoice = "Type3";
        }

        // Load SPCP similarity threshold from config (same value SPCPAnalysis uses)
        try {
            LoadParam lp = new LoadParam();
            this.spcpSimThreshold = lp.getSpcpSimilarity();
        } catch (Exception e) {
            System.out.println("Warning: Could not load SPCP config, using default threshold " + spcpSimThreshold);
        }
    }

    public void print(Object o) {
        System.out.println(o);
    }

    /**
     * Try to find commit_logs.txt in common locations and return the chosen path,
     * or null if not found.
     */
    private String findCommitLogPath() {
        // Detect subject system name (e.g., Ctags)
        String subject = cp.getSubjectSystem();
        LoadParam lp = new LoadParam();

        // Try 1: Code directory (where source revisions are stored)
        String codeDir = lp.getdir();
        if (codeDir != null && !codeDir.isEmpty()) {
            String p0 = codeDir + "/commit_logs.txt";
            if (Files.exists(Paths.get(p0))) {
                System.out.println("Using commit log: " + Paths.get(p0).toAbsolutePath());
                return p0;
            }
        }

        // Try 2: Clone directory (NiCad output location)
        String cloneDir = lp.getCloneDir();
        if (cloneDir != null && !cloneDir.isEmpty()) {
            String p0 = cloneDir + "/commit_logs.txt";
            if (Files.exists(Paths.get(p0))) {
                System.out.println("Using commit log: " + Paths.get(p0).toAbsolutePath());
                return p0;
            }
        }

        // Try 3: WorkFolder/{subjectSystem}/commit_logs.txt
        String p1 = "WorkFolder/" + subject + "/commit_logs.txt";
        String p2 = "./WorkFolder/" + subject + "/commit_logs.txt";

        if (Files.exists(Paths.get(p1))) {
            System.out.println("Using commit log: " + Paths.get(p1).toAbsolutePath());
            return p1;
        }

        if (Files.exists(Paths.get(p2))) {
            System.out.println("Using commit log: " + Paths.get(p2).toAbsolutePath());
            return p2;
        }

        System.out.println("ERROR: commit_logs.txt not found in:");
        System.out.println("  - Code Dir: " + codeDir);
        System.out.println("  - Clone Dir: " + cloneDir);
        System.out.println("  - WorkFolder/" + subject);
        return null;
    }

    /**
     * Parse commit_logs.txt and produce a mapping from revision -> author,
     * filling missing revisions by carrying forward last seen author.
     */
    private Map<Integer, String> buildRevisionAuthorMap(String commitFile, int minRev, int maxRev) {
        Map<Integer, String> logAuthors = new TreeMap<>(); // sorted by rev

        if (commitFile == null) {
            // No commit file found — return a map filled with UNKNOWN
            Map<Integer, String> unknownMap = new HashMap<>();
            for (int r = minRev; r <= maxRev; r++)
                unknownMap.put(r, "UNKNOWN");
            return unknownMap;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(commitFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty())
                    continue;

                // Expect: revision | filepath | change type | line number | commit message |
                // author | date
                String[] parts = line.split("\\|");
                if (parts.length < 6)
                    continue; // skip malformed
                String revStr = parts[0].trim();
                String author = parts[5].trim();

                try {
                    int rev = Integer.parseInt(revStr);
                    logAuthors.put(rev, author);
                } catch (NumberFormatException nfe) {
                    // skip header or malformed line
                }
            }
        } catch (IOException e) {
            System.out.println("Failed to read commit logs: " + e.getMessage());
            // fallback: unknown authors map
            Map<Integer, String> unknownMap = new HashMap<>();
            for (int r = minRev; r <= maxRev; r++)
                unknownMap.put(r, "UNKNOWN");
            return unknownMap;
        }

        // Expand to all revisions by carrying forward lastAuthor
        Map<Integer, String> revAuthor = new HashMap<>();
        String lastAuthor = null;
        SortedSet<Integer> knownRevs = new TreeSet<>(logAuthors.keySet());
        // If there is no entry <= minRev, try to set lastAuthor to first known author
        if (!knownRevs.isEmpty()) {
            Integer firstKnown = knownRevs.first();
            if (firstKnown <= minRev) {
                lastAuthor = logAuthors.get(firstKnown);
            } else {
                // If commit log starts after minRev, carry its author backwards as well
                lastAuthor = logAuthors.get(firstKnown);
            }
        }

        for (int r = minRev; r <= maxRev; r++) {
            if (logAuthors.containsKey(r)) {
                lastAuthor = logAuthors.get(r);
            }
            if (lastAuthor == null) {
                revAuthor.put(r, "UNKNOWN");
            } else {
                revAuthor.put(r, lastAuthor);
            }
        }

        return revAuthor;
    }

    private String extractFileName(String fullPath) {
        if (fullPath == null)
            return "";
        fullPath = fullPath.trim();
        int idx = fullPath.lastIndexOf('/');
        if (idx == -1)
            idx = fullPath.lastIndexOf('\\');
        return (idx == -1) ? fullPath : fullPath.substring(idx + 1);
    }

    /**
     * Main method to compute clone lifecycle and produce CSV outputs.
     */
    public void computeCloneLifecycle() throws IOException {
        // Map to store clone histories by globalCloneId
        Map<Integer, CloneHistory> historyMap = new HashMap<>();

        // Map to store clone info (including classId & filepath) for pair generation
        Map<Integer, Clones> cloneInfoMap = new HashMap<>();

        // Map for gcid -> filepath (first-seen)
        Map<Integer, String> gcidToFilepath = new HashMap<>();

        // Queue to track active clones
        Queue<Integer> activeClones = new LinkedList<>();

        System.out.println("Starting clone genealogy analysis from revision " + startRev + " to " + endRev);

        // find commit log and build revision->author map
        String commitPath = findCommitLogPath();
        Map<Integer, String> revAuthorMap = buildRevisionAuthorMap(commitPath, startRev, endRev);

        // Step 1: Seed from the start revision (startRev)
        System.out.println("Loading clones from first revision: " + startRev);
        List<Clones> firstRevClones = ap.getGlobalClones(startRev, cChoice, granularity);

        for (Clones c : firstRevClones) {
            int gcid = c.globalCloneId;
            cloneInfoMap.put(gcid, c);
            gcidToFilepath.put(gcid, extractFileName(c.filePath));

            CloneHistory h = new CloneHistory();
            h.globalCloneId = gcid;
            h.addedInRev = startRev;
            h.startRev = startRev;
            h.endRev = startRev;

            if ("M".equalsIgnoreCase(c.cloneType)) {
                h.changeRevs.add(startRev);
            } else if ("U".equalsIgnoreCase(c.cloneType)) {
                h.unchangedRevs.add(startRev);
            }

            historyMap.put(gcid, h);
            activeClones.offer(gcid);
        }

        System.out.println("Found " + activeClones.size() + " clones in first revision");

        // Step 2: Process subsequent revisions and detect newly introduced clones
        for (int rev = startRev + 1; rev <= endRev; rev++) {
            System.out.println("Processing revision: " + rev);

            List<Clones> currClones = ap.getGlobalClones(rev, cChoice, granularity);

            // Map for quick lookup by gcid
            Map<Integer, Clones> currentRevMap = new HashMap<>();
            for (Clones c : currClones) {
                currentRevMap.put(c.globalCloneId, c);
            }

            // nextQueue will hold clones active in the next revision
            Queue<Integer> nextQueue = new LinkedList<>();

            // === NEW clones: those in currentRevMap but not in historyMap ===
            for (Clones c : currClones) {
                int gcid = c.globalCloneId;
                if (!historyMap.containsKey(gcid)) {
                    // New clone introduced at this revision
                    CloneHistory h = new CloneHistory();
                    h.globalCloneId = gcid;
                    h.addedInRev = rev;
                    h.startRev = rev;
                    h.endRev = rev;

                    if ("M".equalsIgnoreCase(c.cloneType)) {
                        h.changeRevs.add(rev);
                    } else if ("U".equalsIgnoreCase(c.cloneType)) {
                        h.unchangedRevs.add(rev);
                    }

                    historyMap.put(gcid, h);
                    cloneInfoMap.put(gcid, c);
                    gcidToFilepath.put(gcid, extractFileName(c.filePath));
                    nextQueue.offer(gcid);

                    System.out.println(
                            "  NEW clone detected at rev " + rev + " -> gcid " + gcid + ", classId " + c.classId);
                }
            }

            // Update existing active clones
            while (!activeClones.isEmpty()) {
                int gcid = activeClones.poll();
                CloneHistory h = historyMap.get(gcid);

                if (currentRevMap.containsKey(gcid)) {
                    // still exists in this revision
                    Clones currentClone = currentRevMap.get(gcid);
                    h.endRev = rev;

                    if ("M".equalsIgnoreCase(currentClone.cloneType)) {
                        h.changeRevs.add(rev);
                    } else if ("U".equalsIgnoreCase(currentClone.cloneType)) {
                        h.unchangedRevs.add(rev);
                    }

                    nextQueue.offer(gcid);
                } else {
                    // deleted/absent in this revision
                    h.deletedInRev = rev;
                    System.out.println("  Clone " + gcid + " deleted at revision " + rev);
                }
            }

            // set up for next iteration
            activeClones = nextQueue;
            System.out.println("  Active clones remaining: " + activeClones.size());
        }

        // Step 3: Mark remaining clones as surviving till endRev
        while (!activeClones.isEmpty()) {
            int gcid = activeClones.poll();
            CloneHistory h = historyMap.get(gcid);
            h.deletedInRev = 0; // 0 means still alive
        }

        System.out.println("Total clones tracked: " + historyMap.size());

        // Step 4: Compute evolution metrics for forecasting
        Map<Integer, List<Integer>> classToClonesMap = buildClassToClonesMap(cloneInfoMap);
        computeEvolutionMetrics(historyMap, classToClonesMap);
        System.out.println("Evolution metrics computed for all clones");

        // Save individual clone genealogy with authors & filepath
        saveAsCSVWithAuthors(historyMap, gcidToFilepath, revAuthorMap);

        // Save class groups and pairs
        saveCloneClassGroups(cloneInfoMap);
        saveClonePairs(cloneInfoMap, historyMap, revAuthorMap, gcidToFilepath);
    }

    /**
     * Build a mapping from classId to list of globalCloneIds
     */
    private Map<Integer, List<Integer>> buildClassToClonesMap(Map<Integer, Clones> cloneInfoMap) {
        Map<Integer, List<Integer>> classToClonesMap = new HashMap<>();
        for (Map.Entry<Integer, Clones> entry : cloneInfoMap.entrySet()) {
            int gcid = entry.getKey();
            int classId = entry.getValue().classId;
            classToClonesMap.computeIfAbsent(classId, k -> new ArrayList<>()).add(gcid);
        }
        return classToClonesMap;
    }

    /**
     * Compute evolution forecasting metrics for all clones
     * Based on Dr. Mondal's methodology for forecasting independent evolution
     */
    private void computeEvolutionMetrics(Map<Integer, CloneHistory> historyMap,
            Map<Integer, List<Integer>> classToClonesMap) {

        // Build reverse lookup: gcid -> classId
        Map<Integer, Integer> gcidToClassId = new HashMap<>();
        for (Map.Entry<Integer, List<Integer>> entry : classToClonesMap.entrySet()) {
            for (int gcid : entry.getValue()) {
                gcidToClassId.put(gcid, entry.getKey());
            }
        }

        for (Map.Entry<Integer, CloneHistory> entry : historyMap.entrySet()) {
            int gcid = entry.getKey();
            CloneHistory h = entry.getValue();

            // 1. Calculate stability index and change-proneness
            calculateStabilityMetrics(h);

            // 2. Calculate independent evolution score
            Integer classId = gcidToClassId.get(gcid);
            if (classId != null) {
                List<Integer> classPeers = classToClonesMap.get(classId);
                calculateIndependentEvolutionScore(h, historyMap, classPeers);
            } else {
                // No peers, mark as highly independent
                h.independentEvolutionScore = 1.0;
                h.independentChangeCount = h.changeRevs.size();
                h.coChangeCount = 0;
            }

            // 3. Detect late propagation
            h.latePropagationStatus = "NONE"; // Will be set by SPCP analysis later
        }
    }

    /**
     * Calculate stability index and change-proneness
     * stabilityIndex = unchangedRevs / totalRevisions (higher = more stable)
     * changeProneness = changedRevs / totalRevisions (higher = more change-prone)
     */
    private void calculateStabilityMetrics(CloneHistory h) {
        int totalRevisions = h.endRev - h.startRev + 1;
        if (totalRevisions <= 0) {
            h.stabilityIndex = 0.0;
            h.changeProneness = 0.0;
            return;
        }
        h.stabilityIndex = (double) h.unchangedRevs.size() / totalRevisions;
        h.changeProneness = (double) h.changeRevs.size() / totalRevisions;
    }

    /**
     * Calculate independent evolution score
     * Measures how often a clone changes independently (not with its class peers)
     * Higher score = more likely to evolve independently
     */
    private void calculateIndependentEvolutionScore(CloneHistory h,
            Map<Integer, CloneHistory> historyMap,
            List<Integer> classPeers) {
        if (classPeers == null || classPeers.size() <= 1) {
            // Only one clone in class - always independent
            h.independentEvolutionScore = 1.0;
            h.independentChangeCount = h.changeRevs.size();
            h.coChangeCount = 0;
            return;
        }

        int independentChanges = 0;
        int coChanges = 0;

        for (int changeRev : h.changeRevs) {
            boolean peerAlsoChanged = false;

            // Check if any peer also changed in the same revision
            for (int peerGcid : classPeers) {
                if (peerGcid == h.globalCloneId)
                    continue; // skip self

                CloneHistory peerH = historyMap.get(peerGcid);
                if (peerH != null && peerH.changeRevs.contains(changeRev)) {
                    peerAlsoChanged = true;
                    break;
                }
            }

            if (peerAlsoChanged) {
                coChanges++;
            } else {
                independentChanges++;
            }
        }

        h.coChangeCount = coChanges;
        h.independentChangeCount = independentChanges;

        int totalChanges = h.changeRevs.size();
        if (totalChanges == 0) {
            h.independentEvolutionScore = 0.0; // No changes = no evolution
        } else {
            h.independentEvolutionScore = (double) independentChanges / totalChanges;
        }
    }

    private void saveAsCSVWithAuthors(Map<Integer, CloneHistory> map,
            Map<Integer, String> gcidToFilepath,
            Map<Integer, String> revAuthorMap) throws IOException {

        String outputPath = cp.getGenealogyPath(cChoice, granularity);
        BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath));

        // Header with evolution forecasting metrics
        bw.write(
                "cloneType,granularity,globalCloneId,filepath,authors,addedInRev,deletedInRev,lifespan," +
                        "totalChanges,totalUnchanged,stabilityIndex,changeProneness,independentEvolutionScore," +
                        "coChangeCount,independentChangeCount,changeRevs,unchangedRevs\n");

        List<CloneHistory> list = new ArrayList<>(map.values());
        list.sort(Comparator.comparingInt(h -> h.globalCloneId));

        for (CloneHistory h : list) {
            String changeStr = "\"" +
                    h.changeRevs.stream().map(String::valueOf).collect(Collectors.joining(", ")) + "\"";

            String unchangedStr = "\"" +
                    h.unchangedRevs.stream().map(String::valueOf).collect(Collectors.joining(", ")) + "\"";

            int deleted = h.deletedInRev > 0 ? h.deletedInRev : h.endRev;
            int lifespan = deleted - h.addedInRev;

            // collect authors from both change and unchanged revisions
            Set<String> authors = new LinkedHashSet<>();
            for (int rev : h.changeRevs) {
                String a = revAuthorMap.get(rev);
                authors.add(a != null ? a : "UNKNOWN");
            }
            for (int rev : h.unchangedRevs) {
                String a = revAuthorMap.get(rev);
                authors.add(a != null ? a : "UNKNOWN");
            }
            // also include addedInRev (in case neither list contains it)
            if (!authors.contains(revAuthorMap.get(h.addedInRev))) {
                String a = revAuthorMap.get(h.addedInRev);
                authors.add(a != null ? a : "UNKNOWN");
            }

            String authorStr = "\"" + String.join(", ", authors) + "\"";

            String filepath = gcidToFilepath.getOrDefault(h.globalCloneId, "");

            bw.write(
                    cChoice + "," +
                            granularity + "," +
                            h.globalCloneId + "," +
                            filepath + "," +
                            authorStr + "," +
                            h.addedInRev + "," +
                            (h.deletedInRev > 0 ? h.deletedInRev : "Still Active") + "," +
                            lifespan + "," +
                            h.changeRevs.size() + "," +
                            h.unchangedRevs.size() + "," +
                            String.format("%.4f", h.stabilityIndex) + "," +
                            String.format("%.4f", h.changeProneness) + "," +
                            String.format("%.4f", h.independentEvolutionScore) + "," +
                            h.coChangeCount + "," +
                            h.independentChangeCount + "," +
                            changeStr + "," +
                            unchangedStr +
                            "\n");
        }

        bw.close();
        System.out.println("CSV saved: " + outputPath);
        System.out.println("Total clones written: " + list.size());
    }

    private void saveClonePairs(Map<Integer, Clones> cloneInfoMap, Map<Integer, CloneHistory> historyMap,
            Map<Integer, String> revAuthorMap, Map<Integer, String> gcidToFilepath) throws IOException {
        // Group clones by classId
        Map<Integer, List<Integer>> classToClonesMap = new HashMap<>();

        for (Map.Entry<Integer, Clones> entry : cloneInfoMap.entrySet()) {
            int gcid = entry.getKey();
            int classId = entry.getValue().classId;

            classToClonesMap.computeIfAbsent(classId, k -> new ArrayList<>()).add(gcid);
        }

        // Generate pairs
        List<ClonePair> pairs = new ArrayList<>();
        int pairId = 1;

        for (Map.Entry<Integer, List<Integer>> entry : classToClonesMap.entrySet()) {
            int classId = entry.getKey();
            List<Integer> clones = entry.getValue();

            // Create pairs from clones with same classId
            for (int i = 0; i < clones.size(); i++) {
                for (int j = i + 1; j < clones.size(); j++) {
                    int gcid1 = clones.get(i);
                    int gcid2 = clones.get(j);

                    CloneHistory h1 = historyMap.get(gcid1);
                    CloneHistory h2 = historyMap.get(gcid2);
                    Clones c1 = cloneInfoMap.get(gcid1);
                    Clones c2 = cloneInfoMap.get(gcid2);

                    if (h1 != null && h2 != null && c1 != null && c2 != null) {
                        ClonePair pair = new ClonePair();
                        pair.pairId = pairId++;
                        pair.classId = classId;
                        pair.clone1_gcid = gcid1;
                        pair.clone2_gcid = gcid2;
                        pair.clone1_addedRev = h1.addedInRev;
                        pair.clone2_addedRev = h2.addedInRev;
                        pair.clone1_deletedRev = h1.deletedInRev;
                        pair.clone2_deletedRev = h2.deletedInRev;
                        pair.clone1_lifespan = (h1.deletedInRev > 0 ? h1.deletedInRev : h1.endRev) - h1.addedInRev;
                        pair.clone2_lifespan = (h2.deletedInRev > 0 ? h2.deletedInRev : h2.endRev) - h2.addedInRev;

                        // Calculate co-change and independent change counts
                        int coChange = 0;
                        int gcid1IndependentChange = 0;
                        int gcid2IndependentChange = 0;

                        // Find overlapping revision range
                        int pairStartRev = Math.max(h1.addedInRev, h2.addedInRev);
                        int pairEndRev = Math.min(
                                h1.deletedInRev > 0 ? h1.deletedInRev : h1.endRev,
                                h2.deletedInRev > 0 ? h2.deletedInRev : h2.endRev);

                        for (int rev = pairStartRev; rev <= pairEndRev; rev++) {
                            boolean changed1 = h1.changeRevs.contains(rev);
                            boolean changed2 = h2.changeRevs.contains(rev);

                            if (changed1 && changed2) {
                                coChange++;
                            } else if (changed1 && !changed2) {
                                gcid1IndependentChange++;
                            } else if (!changed1 && changed2) {
                                gcid2IndependentChange++;
                            }
                        }

                        // Collect ALL authors across the entire lifecycle of each gcid
                        // (from both changed and unchanged revisions, plus addedInRev)
                        Set<String> gcid1Authors = new LinkedHashSet<>();
                        Set<String> gcid2Authors = new LinkedHashSet<>();

                        // Collect from changeRevs
                        for (int rev : h1.changeRevs) {
                            String author = revAuthorMap.get(rev);
                            gcid1Authors.add(author != null ? author : "UNKNOWN");
                        }
                        for (int rev : h2.changeRevs) {
                            String author = revAuthorMap.get(rev);
                            gcid2Authors.add(author != null ? author : "UNKNOWN");
                        }

                        // Collect from unchangedRevs
                        for (int rev : h1.unchangedRevs) {
                            String author = revAuthorMap.get(rev);
                            gcid1Authors.add(author != null ? author : "UNKNOWN");
                        }
                        for (int rev : h2.unchangedRevs) {
                            String author = revAuthorMap.get(rev);
                            gcid2Authors.add(author != null ? author : "UNKNOWN");
                        }

                        // Also include addedInRev (in case neither list contains it)
                        String author1 = revAuthorMap.get(h1.addedInRev);
                        gcid1Authors.add(author1 != null ? author1 : "UNKNOWN");
                        String author2 = revAuthorMap.get(h2.addedInRev);
                        gcid2Authors.add(author2 != null ? author2 : "UNKNOWN");

                        pair.coChange = coChange;
                        pair.gcid1IndependentChange = gcid1IndependentChange;
                        pair.gcid2IndependentChange = gcid2IndependentChange;
                        pair.similarity = c1.similarity; // Use similarity from class (both clones share same class)
                        pair.filepath1 = extractFileName(c1.filePath);
                        pair.filepath2 = extractFileName(c2.filePath);
                        pair.fileDistance = calculatePathDepth(c1.filePath, c2.filePath);
                        pair.gcid1Authors = String.join("; ", gcid1Authors);
                        pair.gcid2Authors = String.join("; ", gcid2Authors);

                        // Time-decay weighted coupling features
                        pair.weightedCouplingStrength = computeWeightedCouplingStrength(
                                h1, h2, pairStartRev, pairEndRev);
                        pair.couplingTrend = computeCouplingTrend(h1, h2, pairStartRev, pairEndRev);
                        pair.lastCoChangeAge = lastCoChangeAge(h1, h2, pairEndRev);
                        pair.lastSoloChangeAge = lastSoloChangeAge(h1, h2, pairEndRev);

                        pairs.add(pair);
                    }
                }
            }
        }

        // Save pairs to CSV
        String pairOutputPath = cp.getGenealogyPath(cChoice, granularity).replace(".csv", "_pairs.csv");
        BufferedWriter bw = new BufferedWriter(new FileWriter(pairOutputPath));

        // Load method info for pairs CSV enrichment
        Map<Integer, CloneMethodInfo> pairMethodInfo = loadMethodInfoFromGlobalCloneInfo();

        bw.write(
                "pairId,cloneType,granularity,classId,clone1_gcid,clone2_gcid,clone1_addedRev,clone2_addedRev," +
                        "clone1_deletedRev,clone2_deletedRev,clone1_lifespan,clone2_lifespan," +
                        "co_change,gcid1_independent_change,gcid2_independent_change,similarity," +
                        "filepath1,filepath2,file_distance,gcid1_authors,gcid2_authors," +
                        "weighted_coupling_strength,coupling_trend,last_co_change_age,last_solo_change_age," +
                        "method_name1,method_signature1,method_name2,method_signature2\n");

        for (ClonePair pair : pairs) {
            // Look up method info for each clone in this pair
            CloneMethodInfo mi1 = pairMethodInfo.get(pair.clone1_gcid);
            CloneMethodInfo mi2 = pairMethodInfo.get(pair.clone2_gcid);
            String pMn1  = (mi1 != null && mi1.methodName      != null) ? mi1.methodName      : "NULL";
            String pMs1  = (mi1 != null && mi1.methodSignature != null) ? mi1.methodSignature : "NULL";
            String pMn2  = (mi2 != null && mi2.methodName      != null) ? mi2.methodName      : "NULL";
            String pMs2  = (mi2 != null && mi2.methodSignature != null) ? mi2.methodSignature : "NULL";

            bw.write(
                    pair.pairId + "," +
                            cChoice + "," +
                            granularity + "," +
                            pair.classId + "," +
                            pair.clone1_gcid + "," +
                            pair.clone2_gcid + "," +
                            pair.clone1_addedRev + "," +
                            pair.clone2_addedRev + "," +
                            (pair.clone1_deletedRev > 0 ? pair.clone1_deletedRev : "Still Active") + "," +
                            (pair.clone2_deletedRev > 0 ? pair.clone2_deletedRev : "Still Active") + "," +
                            pair.clone1_lifespan + "," +
                            pair.clone2_lifespan + "," +
                            pair.coChange + "," +
                            pair.gcid1IndependentChange + "," +
                            pair.gcid2IndependentChange + "," +
                            pair.similarity + "," +
                            quote(pair.filepath1) + "," +
                            quote(pair.filepath2) + "," +
                            pair.fileDistance + "," +
                            quote(pair.gcid1Authors) + "," +
                            quote(pair.gcid2Authors) + "," +
                            String.format("%.4f", pair.weightedCouplingStrength) + "," +
                            String.format("%.4f", pair.couplingTrend) + "," +
                            pair.lastCoChangeAge + "," +
                            pair.lastSoloChangeAge + "," +
                            quote(pMn1) + "," +
                            quote(pMs1) + "," +
                            quote(pMn2) + "," +
                            quote(pMs2) +
                            "\n");
        }

        bw.close();
        System.out.println("Clone Pairs CSV saved: " + pairOutputPath);
        System.out.println("Total pairs written: " + pairs.size());
    }

    private void saveCloneClassGroups(Map<Integer, Clones> cloneInfoMap) throws IOException {

        // Group clone GCIDs by classId
        Map<Integer, List<Integer>> classToClonesMap = new HashMap<>();

        for (Map.Entry<Integer, Clones> entry : cloneInfoMap.entrySet()) {
            int gcid = entry.getKey();
            int classId = entry.getValue().classId;

            classToClonesMap.computeIfAbsent(classId, k -> new ArrayList<>()).add(gcid);
        }

        // Output path
        String outputPath = cp.getGenealogyPath(cChoice, granularity)
                .replace(".csv", "_classGroups.csv");

        BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath));

        // Header
        bw.write("classId,clone_gcid_list\n");

        // Write each class → list of clones
        for (Map.Entry<Integer, List<Integer>> entry : classToClonesMap.entrySet()) {
            int classId = entry.getKey();
            List<Integer> ids = entry.getValue();

            // Sort IDs for nice readable output
            Collections.sort(ids);

            // Convert to [1,2,3,4]
            String listStr = ids.toString();

            bw.write(classId + "," + listStr + "\n");
        }

        bw.close();
        System.out.println("Clone Class Groups saved: " + outputPath);
    }

    // Helper class for clone pairs
    private static class ClonePair {
        int pairId;
        int classId;
        int clone1_gcid;
        int clone2_gcid;
        int clone1_addedRev;
        int clone2_addedRev;
        int clone1_deletedRev;
        int clone2_deletedRev;
        int clone1_lifespan;
        int clone2_lifespan;
        // Raw co-change counts
        int coChange;
        int gcid1IndependentChange;
        int gcid2IndependentChange;
        int similarity;
        String filepath1;
        String filepath2;
        int fileDistance;
        String gcid1Authors;
        String gcid2Authors;
        // Time-decay weighted coupling features
        double weightedCouplingStrength; // WCS at end of pair lifetime
        double couplingTrend;            // recent_WCS - early_WCS (negative = decoupling)
        int lastCoChangeAge;             // revisions since last co-change (-1 if never)
        int lastSoloChangeAge;           // revisions since last solo change (-1 if never)
    }

    // ============== EVOLUTION FORECASTING DATASET EXPORT ==============

    /**
     * Data structure for evolution dataset records
     * Each record represents a clone pair's evolution state at a specific revision
     */
    private static class EvolutionRecord {
        int revision;
        int gcid1, gcid2;
        int classId;
        int coChangeCount; // cumulative co-changes up to this revision
        int gcid1IndependentCount; // cumulative independent changes for gcid1
        int gcid2IndependentCount; // cumulative independent changes for gcid2
        String filepath1, filepath2;
        String author1, author2;
        int similarity;
        // Additional forecasting parameters
        int nlines1, nlines2;
        int depth; // folder distance: 0 = same file, higher = more folder hops
        int classSize;
        String cloneType;
        boolean isSPCP; // true if this gcid pair exists in SPCP analysis
        // Method info for each gcid
        String methodName1, methodSignature1;
        String methodName2, methodSignature2;
        // Time-decay weighted coupling features (as of this revision)
        double weightedCouplingStrength;
        int lastCoChangeAge;   // revisions since last co-change up to this rev (-1 if never)
        int lastSoloChangeAge; // revisions since last solo change up to this rev (-1 if never)
        double couplingTrend;              // WCS(recent half) - WCS(early half); negative = decoupling
        double wcsRecent;                  // WCS over last WCS_RECENT_WINDOW revisions only
        int soloEventsAfterLastCoChange;   // solo changes (either clone) since last co-change
        // Binary ML-ready flags (derived from raw strings)
        int sameFile;   // 1 if filepath1 == filepath2
        int sameAuthor; // 1 if author1 == author2
        int sameMethod; // 1 if both clones are in the same method (non-null matching names)
        // --- NEW PROCESS METRICS ---
        int noc1, noc2;               // Number of Changes per clone
        int fileAge1, fileAge2;        // Clone lifetime in revisions
        int churn1, churn2;            // True lines changed inside clone region
        // --- NEW OWNERSHIP METRICS ---
        int distinctAuthors1, distinctAuthors2;
        double majorAuthorProp1, majorAuthorProp2; // TCO proportion
        int minorAuthorCount1, minorAuthorCount2;  // Authors with < 5% of commits
        // --- WIES + TARGET LABEL ---
        double wies;                      // Weighted Independent Evolution Score = 1 - WCS
        int will_independently_evolve;    // 1 if wies > 0.5, else 0
        // --- DECAY-WEIGHTED METRICS (5 half-lives × 3 metrics = 15 features) ---
        double[] irDecay  = new double[5]; // ir_decay at h={10,20,30,50,75}
        double[] csDecay  = new double[5]; // cs_decay at h={10,20,30,50,75}
        double[] spcpDecay = new double[5]; // spcp_decay at h={10,20,30,50,75}
        // --- COMPOSITE DECAY-WEIGHTED LABEL (Option C) ---
        int willDivergeDecay;             // 1 if composite hierarchy says independent (using h=20)
    }

    /**
     * Data structure to hold method info for a clone
     */
    private static class CloneMethodInfo {
        String methodName;
        String methodSignature;
    }

    /**
     * Export evolution dataset for forecasting independent clone evolution.
     * Output format:
     * Revision | gcid1 | gcid2 | classid | co_change_count |
     * gcid1_independent_change_count |
     * gcid2_independent_change_count | filepath1 | filepath2 | author1 | author2 |
     * similarity |
     * clone_type | nlines1 | nlines2 | same_file | class_size
     * 
     * Only includes rows where co_change_count > 0 OR any independent_change_count
     * > 0
     */
    public void exportEvolutionDataset() throws IOException {
        System.out.println("=== Starting Evolution Dataset Export ===");
        System.out.println("Clone Type: " + cChoice + ", Granularity: " + granularity);
        System.out.println("Revision range: " + startRev + " to " + endRev);

        // Build revision -> author map from commit logs
        String commitPath = findCommitLogPath();
        Map<Integer, String> revAuthorMap = buildRevisionAuthorMap(commitPath, startRev, endRev);

        // Load SPCP pairs for this clone type and granularity
        Set<String> spcpPairs = loadSPCPPairs();

        // Load method info from GlobalCloneInfo
        Map<Integer, CloneMethodInfo> gcidMethodInfo = loadMethodInfoFromGlobalCloneInfo();

        // Maps to track clone information across revisions
        Map<Integer, Clones> cloneInfoMap = new HashMap<>(); // gcid -> latest Clones info
        Map<Integer, String> gcidToFilename = new HashMap<>(); // gcid -> filename (for display)
        Map<Integer, String> gcidToFullPath = new HashMap<>(); // gcid -> full filepath (for depth calc)
        Map<Integer, CloneHistory> historyMap = new HashMap<>(); // gcid -> CloneHistory

        // Track which clones exist at each revision and their change status
        Map<Integer, Map<Integer, String>> revisionCloneStatus = new HashMap<>(); // rev -> (gcid -> changeType)

        // Store per-revision clone instances for dynamic SPCP fragment fetching
        // Structure: gcid -> (revision -> Clones instance with filePath, startLine, endLine)
        Map<Integer, Map<Integer, Clones>> cloneInstancesByRev = new HashMap<>();

        System.out.println("Loading clones from all revisions...");

        // Step 1: Load all clones from all revisions to build complete picture
        for (int rev = startRev; rev <= endRev; rev++) {
            List<Clones> clones = ap.getGlobalClones(rev, cChoice, granularity);
            Map<Integer, String> revStatus = new HashMap<>();

            for (Clones c : clones) {
                int gcid = c.globalCloneId;
                revStatus.put(gcid, c.cloneType); // M, U, A, D

                // Store latest clone info
                cloneInfoMap.put(gcid, c);

                // Store filepath (first seen) - both filename and full path
                if (!gcidToFilename.containsKey(gcid)) {
                    gcidToFilename.put(gcid, extractFileName(c.filePath));
                    gcidToFullPath.put(gcid, c.filePath);
                }

                // Initialize or update history
                CloneHistory h = historyMap.get(gcid);
                if (h == null) {
                    h = new CloneHistory();
                    h.globalCloneId = gcid;
                    h.addedInRev = rev;
                    h.startRev = rev;
                    historyMap.put(gcid, h);
                }
                h.endRev = rev;

                if ("M".equalsIgnoreCase(c.cloneType)) {
                    h.changeRevs.add(rev);
                } else if ("U".equalsIgnoreCase(c.cloneType)) {
                    h.unchangedRevs.add(rev);
                }

                // Store clone instance for this revision (for SPCP fragment fetching)
                cloneInstancesByRev.computeIfAbsent(gcid, k -> new HashMap<>()).put(rev, c);
            }
            revisionCloneStatus.put(rev, revStatus);
        }

        System.out.println("Total unique clones found: " + cloneInfoMap.size());

        // Step 1b: Compute process + ownership metrics (uses historyMap + revAuthorMap)
        computeProcessAndOwnershipMetrics(historyMap, revAuthorMap, gcidToFilename);

        // Step 2: Group clones by classId to identify pairs
        Map<Integer, List<Integer>> classToClonesMap = new HashMap<>();
        for (Map.Entry<Integer, Clones> entry : cloneInfoMap.entrySet()) {
            int gcid = entry.getKey();
            int classId = entry.getValue().classId;
            classToClonesMap.computeIfAbsent(classId, k -> new ArrayList<>()).add(gcid);
        }

        System.out.println("Total clone classes: " + classToClonesMap.size());

        // Step 3: Generate evolution records for each clone pair
        List<EvolutionRecord> records = new ArrayList<>();

        for (Map.Entry<Integer, List<Integer>> classEntry : classToClonesMap.entrySet()) {
            int classId = classEntry.getKey();
            List<Integer> classClones = classEntry.getValue();
            int classSize = classClones.size();

            if (classSize < 2) {
                continue; // Need at least 2 clones to form a pair
            }

            // Generate all pairs within this class
            for (int i = 0; i < classClones.size(); i++) {
                for (int j = i + 1; j < classClones.size(); j++) {
                    int gcid1 = classClones.get(i);
                    int gcid2 = classClones.get(j);

                    Clones c1 = cloneInfoMap.get(gcid1);
                    Clones c2 = cloneInfoMap.get(gcid2);
                    CloneHistory h1 = historyMap.get(gcid1);
                    CloneHistory h2 = historyMap.get(gcid2);

                    if (c1 == null || c2 == null || h1 == null || h2 == null) {
                        continue;
                    }

                    // Pre-compute SPCP-verified revisions for this pair (once, not per event)
                    // This set grows as we iterate revisions — but we compute it up to pairEndRev
                    // so it covers the entire pair lifetime. Per-event SPCP_decay uses only
                    // the subset of spcpRevisions up to the current revision.
                    int pairEndRev = Math.min(h1.endRev, h2.endRev);
                    Set<Integer> spcpRevisions = computeSPCPRevisions(
                            gcid1, gcid2, h1, h2, pairEndRev, cloneInstancesByRev);

                    // Track cumulative counts
                    int coChangeCount = 0;
                    int gcid1IndependentCount = 0;
                    int gcid2IndependentCount = 0;

                    // Analyze each revision where at least one clone exists
                    int pairStartRev = Math.max(h1.addedInRev, h2.addedInRev);

                    for (int rev = pairStartRev; rev <= pairEndRev; rev++) {
                        Map<Integer, String> revStatus = revisionCloneStatus.get(rev);
                        if (revStatus == null)
                            continue;

                        String status1 = revStatus.get(gcid1);
                        String status2 = revStatus.get(gcid2);

                        boolean changed1 = "M".equalsIgnoreCase(status1);
                        boolean changed2 = "M".equalsIgnoreCase(status2);

                        if (changed1 && changed2) {
                            coChangeCount++;
                        } else if (changed1 && !changed2) {
                            gcid1IndependentCount++;
                        } else if (!changed1 && changed2) {
                            gcid2IndependentCount++;
                        }

                        // Only create record if there's any change activity
                        if (coChangeCount > 0 || gcid1IndependentCount > 0 || gcid2IndependentCount > 0) {
                            // Check if this revision has new change activity
                            if (changed1 || changed2) {
                                EvolutionRecord rec = new EvolutionRecord();
                                rec.revision = rev;
                                rec.gcid1 = gcid1;
                                rec.gcid2 = gcid2;
                                rec.classId = classId;
                                rec.coChangeCount = coChangeCount;
                                rec.gcid1IndependentCount = gcid1IndependentCount;
                                rec.gcid2IndependentCount = gcid2IndependentCount;

                                // File paths (filenames for display)
                                rec.filepath1 = gcidToFilename.getOrDefault(gcid1, "");
                                rec.filepath2 = gcidToFilename.getOrDefault(gcid2, "");

                                // Authors for this revision
                                rec.author1 = revAuthorMap.getOrDefault(rev, "UNKNOWN");
                                rec.author2 = revAuthorMap.getOrDefault(rev, "UNKNOWN");

                                // Similarity (from NiCad, using the class similarity)
                                rec.similarity = c1.similarity;

                                // Additional parameters
                                rec.nlines1 = c1.nLines;
                                rec.nlines2 = c2.nLines;
                                // Calculate folder depth using full paths
                                String fullPath1 = gcidToFullPath.getOrDefault(gcid1, "");
                                String fullPath2 = gcidToFullPath.getOrDefault(gcid2, "");
                                rec.depth = calculatePathDepth(fullPath1, fullPath2);
                                rec.classSize = classSize;
                                rec.cloneType = cChoice;

                                // Check if this pair is in SPCP analysis
                                rec.isSPCP = spcpPairs.contains(gcid1 + "-" + gcid2)
                                        || spcpPairs.contains(gcid2 + "-" + gcid1);

                                // Add method info from GlobalCloneInfo
                                CloneMethodInfo methodInfo1 = gcidMethodInfo.get(gcid1);
                                CloneMethodInfo methodInfo2 = gcidMethodInfo.get(gcid2);
                                rec.methodName1 = methodInfo1 != null ? methodInfo1.methodName : "NULL";
                                rec.methodSignature1 = methodInfo1 != null ? methodInfo1.methodSignature : "NULL";
                                rec.methodName2 = methodInfo2 != null ? methodInfo2.methodName : "NULL";
                                rec.methodSignature2 = methodInfo2 != null ? methodInfo2.methodSignature : "NULL";

                                // Time-decay weighted coupling (as of this revision)
                                rec.weightedCouplingStrength = computeWeightedCouplingStrength(
                                        h1, h2, pairStartRev, rev);
                                rec.lastCoChangeAge = lastCoChangeAge(h1, h2, rev);
                                rec.lastSoloChangeAge = lastSoloChangeAge(h1, h2, rev);

                                // Decoupling signals
                                int midRev = pairStartRev + (rev - pairStartRev) / 2;
                                if (rev - pairStartRev >= 2) {
                                    double earlyWCS  = computeWeightedCouplingStrength(h1, h2, pairStartRev, midRev);
                                    double recentWCS = computeWeightedCouplingStrength(h1, h2, midRev + 1, rev);
                                    rec.couplingTrend = recentWCS - earlyWCS;
                                } else {
                                    rec.couplingTrend = 0.0;
                                }
                                rec.wcsRecent = computeWeightedCouplingStrength(
                                        h1, h2, Math.max(pairStartRev, rev - WCS_RECENT_WINDOW), rev);
                                rec.soloEventsAfterLastCoChange = countSoloAfterLastCoChange(h1, h2, rev);

                                // Binary ML flags derived from raw string fields
                                rec.sameFile = rec.filepath1.equals(rec.filepath2) ? 1 : 0;
                                String a1 = rec.author1 != null ? rec.author1 : "";
                                String a2 = rec.author2 != null ? rec.author2 : "";
                                rec.sameAuthor = a1.equals(a2) && !a1.isEmpty() ? 1 : 0;
                                String mn1 = (methodInfo1 != null && methodInfo1.methodName != null) ? methodInfo1.methodName : "";
                                String mn2 = (methodInfo2 != null && methodInfo2.methodName != null) ? methodInfo2.methodName : "";
                                rec.sameMethod = (!mn1.isEmpty() && mn1.equals(mn2)) ? 1 : 0;

                                // --- Process metrics per clone (from CloneHistory) ---
                                rec.noc1     = h1.noc;     rec.noc2     = h2.noc;
                                rec.fileAge1 = h1.fileAge; rec.fileAge2 = h2.fileAge;
                                rec.churn1   = h1.churn;   rec.churn2   = h2.churn;

                                // --- Ownership metrics per clone ---
                                rec.distinctAuthors1    = h1.distinctAuthors;       rec.distinctAuthors2    = h2.distinctAuthors;
                                rec.majorAuthorProp1    = h1.majorAuthorProportion; rec.majorAuthorProp2    = h2.majorAuthorProportion;
                                rec.minorAuthorCount1   = h1.minorAuthorCount;      rec.minorAuthorCount2   = h2.minorAuthorCount;

                                // --- WIES (time-decay weighted independent evolution score) ---
                                // WIES = 1 - WCS: recent independent changes outweigh older co-changes
                                rec.wies = 1.0 - rec.weightedCouplingStrength;
                                rec.will_independently_evolve = rec.wies > 0.5 ? 1 : 0;

                                // --- DECAY-WEIGHTED METRICS at 5 candidate half-lives ---
                                // Co-change revisions for this pair up to current revision
                                Set<Integer> coChangeRevsUpToNow = new TreeSet<>();
                                for (int r : h1.changeRevs) {
                                    if (r <= rev && h2.changeRevs.contains(r)) {
                                        coChangeRevsUpToNow.add(r);
                                    }
                                }
                                // SPCP revisions filtered to only those up to current revision
                                Set<Integer> spcpRevsUpToNow = new TreeSet<>();
                                for (int r : spcpRevisions) {
                                    if (r <= rev) spcpRevsUpToNow.add(r);
                                }

                                for (int hi = 0; hi < HALF_LIVES.length; hi++) {
                                    double lambda = lambdaFromHalfLife(HALF_LIVES[hi]);
                                    rec.irDecay[hi] = computeDecayWeightedIR(h1, h2, pairStartRev, rev, lambda);
                                    rec.csDecay[hi] = computeDecayWeightedCS(h1, h2, pairStartRev, rev, lambda);
                                    rec.spcpDecay[hi] = computeDecayWeightedSPCP(
                                            spcpRevsUpToNow, coChangeRevsUpToNow, rev, lambda);
                                }

                                // --- Option C composite label using h=20 (index 1) ---
                                // DEPENDENT if CS_decay >= 0.5 AND SPCP_decay >= 0.5 AND IR_decay < 0.5
                                // INDEPENDENT otherwise
                                boolean dependent = rec.csDecay[1] >= 0.5
                                        && rec.spcpDecay[1] >= 0.5
                                        && rec.irDecay[1] < 0.5;
                                rec.willDivergeDecay = dependent ? 0 : 1;

                                records.add(rec);
                            }
                        }
                    }
                }
            }
        }

        System.out.println("Total evolution records generated: " + records.size());

        // Sort records by revision (then by gcid1, gcid2 for consistency)
        records.sort((a, b) -> {
            if (a.revision != b.revision)
                return Integer.compare(a.revision, b.revision);
            if (a.gcid1 != b.gcid1)
                return Integer.compare(a.gcid1, b.gcid1);
            return Integer.compare(a.gcid2, b.gcid2);
        });

        // Step 4: Write to CSV
        String outputPath = cp.getGenealogyPath(cChoice, granularity)
                .replace(".csv", "_evolution_dataset.csv");

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath))) {
            // Write header — 35 columns (20 original + 14 new + 1 target label)
            bw.write("Revision,gcid1,gcid2,classid,co_change_count,gcid1_independent_change_count," +
                    "gcid2_independent_change_count,same_file,same_author,same_method,similarity," +
                    "clone_type,nlines1,nlines2,depth,class_size,is_spcp," +
                    "weighted_coupling_strength,last_co_change_age,last_solo_change_age," +
                    "coupling_trend,wcs_recent,solo_events_after_last_cochange," +
                    "noc1,noc2,file_age1,file_age2,churn1,churn2," +
                    "distinct_authors1,distinct_authors2," +
                    "major_author_prop1,major_author_prop2," +
                    "minor_author_count1,minor_author_count2," +
                    "wies,will_independently_evolve," +
                    "ir_decay_h10,ir_decay_h20,ir_decay_h30,ir_decay_h50,ir_decay_h75," +
                    "cs_decay_h10,cs_decay_h20,cs_decay_h30,cs_decay_h50,cs_decay_h75," +
                    "spcp_decay_h10,spcp_decay_h20,spcp_decay_h30,spcp_decay_h50,spcp_decay_h75," +
                    "will_diverge_decay\n");

            // Write records
            for (EvolutionRecord rec : records) {
                StringBuilder sb = new StringBuilder();
                sb.append(String.format(
                        "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%.4f,%d,%d," +
                        "%.4f,%.4f,%d," +
                        "%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%d,%d,%.4f,%d",
                        rec.revision,
                        rec.gcid1,
                        rec.gcid2,
                        rec.classId,
                        rec.coChangeCount,
                        rec.gcid1IndependentCount,
                        rec.gcid2IndependentCount,
                        rec.sameFile,
                        rec.sameAuthor,
                        rec.sameMethod,
                        rec.similarity,
                        rec.cloneType,
                        rec.nlines1,
                        rec.nlines2,
                        rec.depth,
                        rec.classSize,
                        rec.isSPCP ? 1 : 0,
                        rec.weightedCouplingStrength,
                        rec.lastCoChangeAge,
                        rec.lastSoloChangeAge,
                        // decoupling signals
                        rec.couplingTrend,
                        rec.wcsRecent,
                        rec.soloEventsAfterLastCoChange,
                        // new process metrics
                        rec.noc1,
                        rec.noc2,
                        rec.fileAge1,
                        rec.fileAge2,
                        rec.churn1,
                        rec.churn2,
                        // new ownership metrics
                        rec.distinctAuthors1,
                        rec.distinctAuthors2,
                        rec.majorAuthorProp1,
                        rec.majorAuthorProp2,
                        rec.minorAuthorCount1,
                        rec.minorAuthorCount2,
                        // WIES + target label
                        rec.wies,
                        rec.will_independently_evolve));

                // Append decay-weighted metrics (15 features + 1 label)
                for (int hi = 0; hi < HALF_LIVES.length; hi++) {
                    sb.append(String.format(",%.4f", rec.irDecay[hi]));
                }
                for (int hi = 0; hi < HALF_LIVES.length; hi++) {
                    sb.append(String.format(",%.4f", rec.csDecay[hi]));
                }
                for (int hi = 0; hi < HALF_LIVES.length; hi++) {
                    sb.append(String.format(",%.4f", rec.spcpDecay[hi]));
                }
                sb.append(String.format(",%d", rec.willDivergeDecay));
                sb.append("\n");

                bw.write(sb.toString());
            }
        }

        System.out.println("=== Evolution Dataset Export Complete ===");
        System.out.println("Output file: " + outputPath);
        System.out.println("Total records: " + records.size());
    }

    // ============== REVISION-WISE DATASET EXPORT ==============

    /**
     * Export revision-wise dataset capturing three layers at every revision R:
     *
     *   Layer 1 (fragment): gcid, filePath, changeType, nlines,
     *                        cumulative changes/unchanged, stability, lifespan, author
     *   Layer 2 (pair):     gcid1, gcid2, cumulative co/independent change counts,
     *                        WCS, coupling_trend, last_co_change_age, last_solo_change_age
     *   Layer 3 (fragment←pairs): mean/max/min WCS across all pairs of a gcid,
     *                              count of strongly-coupled and decoupled pairs
     *
     * Output files (Layer 1+3 combined, Layer 2 separate):
     *   _rev_fragment.csv
     *   _rev_pair.csv
     */
    public void exportRevisionWiseDataset() throws IOException {
        System.out.println("=== Starting Revision-Wise Dataset Export ===");
        System.out.println("Clone Type: " + cChoice + ", Granularity: " + granularity);
        System.out.println("Revision range: " + startRev + " to " + endRev);

        String commitPath = findCommitLogPath();
        Map<Integer, String> revAuthorMap = buildRevisionAuthorMap(commitPath, startRev, endRev);

        Map<Integer, Clones>                      cloneInfoMap        = new HashMap<>();
        Map<Integer, String>                      gcidToFilename      = new HashMap<>();
        Map<Integer, String>                      gcidToFullPath      = new HashMap<>();
        Map<Integer, CloneHistory>                historyMap          = new HashMap<>();
        Map<Integer, Map<Integer, Clones>>        cloneInstancesByRev = new HashMap<>();
        Map<Integer, Map<Integer, String>>        revisionCloneStatus = new HashMap<>();

        for (int rev = startRev; rev <= endRev; rev++) {
            List<Clones> clones = ap.getGlobalClones(rev, cChoice, granularity);
            Map<Integer, String> revStatus = new HashMap<>();
            for (Clones c : clones) {
                int gcid = c.globalCloneId;
                revStatus.put(gcid, c.cloneType);
                cloneInfoMap.put(gcid, c);
                gcidToFilename.putIfAbsent(gcid, extractFileName(c.filePath));
                gcidToFullPath.putIfAbsent(gcid, c.filePath);
                CloneHistory h = historyMap.get(gcid);
                if (h == null) {
                    h = new CloneHistory();
                    h.globalCloneId = gcid; h.addedInRev = rev; h.startRev = rev;
                    historyMap.put(gcid, h);
                }
                h.endRev = rev;
                if ("M".equalsIgnoreCase(c.cloneType)) h.changeRevs.add(rev);
                else if ("U".equalsIgnoreCase(c.cloneType)) h.unchangedRevs.add(rev);
                cloneInstancesByRev.computeIfAbsent(gcid, k -> new HashMap<>()).put(rev, c);
            }
            revisionCloneStatus.put(rev, revStatus);
        }

        System.out.println("Total unique clones: " + cloneInfoMap.size());
        computeProcessAndOwnershipMetrics(historyMap, revAuthorMap, gcidToFilename);

        // group clones by class for pair generation
        Map<Integer, List<Integer>> classToClonesMap = new HashMap<>();
        for (Map.Entry<Integer, Clones> e : cloneInfoMap.entrySet())
            classToClonesMap.computeIfAbsent(e.getValue().classId, k -> new ArrayList<>()).add(e.getKey());

        // ----- Filter: drop fragments and pairs with zero change activity -----
        // For forecasting work, fragments that never see an 'M' event and pairs
        // that have no joint change activity carry no signal — exclude them.
        Set<Integer> interestingGcids = new HashSet<>();
        for (Map.Entry<Integer, CloneHistory> e : historyMap.entrySet()) {
            if (!e.getValue().changeRevs.isEmpty()) interestingGcids.add(e.getKey());
        }

        Set<String> interestingPairs = new HashSet<>();
        for (List<Integer> clones : classToClonesMap.values()) {
            for (int i = 0; i < clones.size(); i++) {
                for (int j = i + 1; j < clones.size(); j++) {
                    int g1 = clones.get(i), g2 = clones.get(j);
                    CloneHistory h1 = historyMap.get(g1), h2 = historyMap.get(g2);
                    if (h1 == null || h2 == null) continue;
                    int jointStart = Math.max(h1.addedInRev, h2.addedInRev);
                    int jointEnd   = Math.min(h1.endRev,     h2.endRev);
                    if (jointStart > jointEnd) continue;
                    boolean anyChange = false;
                    for (Integer r : h1.changeRevs)
                        if (r >= jointStart && r <= jointEnd) { anyChange = true; break; }
                    if (!anyChange) for (Integer r : h2.changeRevs)
                        if (r >= jointStart && r <= jointEnd) { anyChange = true; break; }
                    if (anyChange) interestingPairs.add(g1 + "-" + g2);
                }
            }
        }
        System.out.println("Filter: " + interestingGcids.size() + "/" + historyMap.size()
                + " gcids retained (had ≥1 'M' event)");
        System.out.println("Filter: " + interestingPairs.size() + " pairs retained "
                + "(had ≥1 'M' event during joint lifetime)");

        // Load SPCP pairs and pre-compute per-pair SPCP revision sets (for decay features)
        Set<String> spcpPairs = loadSPCPPairs();
        Map<String, Set<Integer>> pairSPCPRevisions = new HashMap<>();
        for (String pk : interestingPairs) {
            String[] pts = pk.split("-");
            int g1 = Integer.parseInt(pts[0]), g2 = Integer.parseInt(pts[1]);
            CloneHistory ph1 = historyMap.get(g1), ph2 = historyMap.get(g2);
            if (ph1 == null || ph2 == null) continue;
            int pEnd = Math.min(ph1.endRev, ph2.endRev);
            pairSPCPRevisions.put(pk, computeSPCPRevisions(g1, g2, ph1, ph2, pEnd, cloneInstancesByRev));
        }

        String basePath = cp.getGenealogyPath(cChoice, granularity).replace(".csv", "");
        String fragCsvPath = basePath + "_rev_fragment.csv";
        String pairCsvPath = basePath + "_rev_pair.csv";

        try (BufferedWriter fragBw = new BufferedWriter(new FileWriter(fragCsvPath));
             BufferedWriter pairBw = new BufferedWriter(new FileWriter(pairCsvPath))) {

            // Layer 1 + Layer 3 header
            fragBw.write(
                "revision,gcid,filePath,changeType,nlines," +
                "totalChanges,totalUnchanged,stabilityIndex,changeProneness,lifespan,author," +
                "activePairs,meanWCS,maxWCS,minWCS,stronglyCoupledPairs,decoupledPairs\n");

            // Layer 2 header
            pairBw.write(
                "revision,gcid1,gcid2,classId,changeType1,changeType2," +
                "coChangeCount,gcid1Independent,gcid2Independent," +
                "weightedCouplingStrength,couplingTrend,lastCoChangeAge,lastSoloChangeAge," +
                "similarity,sameFile,depth," +
                "sameAuthor,classSize,isSpcp,wcsRecent,soloEventsAfterLastCoChange," +
                "nlines1,nlines2,noc1,noc2,fileAge1,fileAge2,churn1,churn2," +
                "distinctAuthors1,distinctAuthors2,majorAuthorProp1,majorAuthorProp2," +
                "minorAuthorCount1,minorAuthorCount2," +
                "ir_decay_h10,ir_decay_h20,ir_decay_h30,ir_decay_h50,ir_decay_h75," +
                "cs_decay_h10,cs_decay_h20,cs_decay_h30,cs_decay_h50,cs_decay_h75," +
                "spcp_decay_h10,spcp_decay_h20,spcp_decay_h30,spcp_decay_h50,spcp_decay_h75\n");

            // Running cumulative counters — avoids rescanning history sets each revision
            Map<Integer, Integer> changesSoFar   = new HashMap<>();
            Map<Integer, Integer> unchangedSoFar = new HashMap<>();
            // Per-pair cumulative [coChange, g1Indep, g2Indep]
            Map<String, int[]> pairCumCounts = new HashMap<>();

            for (int rev = startRev; rev <= endRev; rev++) {
                Map<Integer, String> revStatus =
                        revisionCloneStatus.getOrDefault(rev, Collections.emptyMap());

                // Increment running fragment counters for this revision
                for (Map.Entry<Integer, String> entry : revStatus.entrySet()) {
                    int gcid = entry.getKey();
                    String ct = entry.getValue();
                    if ("M".equalsIgnoreCase(ct))      changesSoFar.merge(gcid, 1, Integer::sum);
                    else if ("U".equalsIgnoreCase(ct)) unchangedSoFar.merge(gcid, 1, Integer::sum);
                }

                // Layer 3 accumulator: gcid -> WCS values from all its active pairs this rev
                Map<Integer, List<Double>> gcidWcsAccum = new HashMap<>();

                // ----- Layer 2: pair coupling rows -----
                for (Map.Entry<Integer, List<Integer>> classEntry : classToClonesMap.entrySet()) {
                    int classId      = classEntry.getKey();
                    List<Integer> gc = classEntry.getValue();

                    for (int i = 0; i < gc.size(); i++) {
                        for (int j = i + 1; j < gc.size(); j++) {
                            int gcid1 = gc.get(i), gcid2 = gc.get(j);

                            // Only emit a row when both clones are alive this revision
                            if (!revStatus.containsKey(gcid1) || !revStatus.containsKey(gcid2)) continue;

                            CloneHistory h1 = historyMap.get(gcid1), h2 = historyMap.get(gcid2);
                            if (h1 == null || h2 == null) continue;

                            // Skip pairs with no joint change activity over their lifetime
                            String pairKey = gcid1 + "-" + gcid2;
                            if (!interestingPairs.contains(pairKey)) continue;

                            // Update cumulative pair counts
                            int[] cum = pairCumCounts.computeIfAbsent(pairKey, k -> new int[3]);
                            String ct1 = revStatus.get(gcid1), ct2 = revStatus.get(gcid2);
                            boolean ch1 = "M".equalsIgnoreCase(ct1), ch2 = "M".equalsIgnoreCase(ct2);
                            if (ch1 && ch2)      cum[0]++;
                            else if (ch1)        cum[1]++;
                            else if (ch2)        cum[2]++;

                            // Only emit rows where at least one clone changed this revision
                            if (!ch1 && !ch2) continue;

                            int pStart = Math.max(h1.addedInRev, h2.addedInRev);
                            double wcs       = computeWeightedCouplingStrength(h1, h2, pStart, rev);
                            double trend     = computeCouplingTrend(h1, h2, pStart, rev);
                            int lcca         = lastCoChangeAge(h1, h2, rev);
                            int lsca         = lastSoloChangeAge(h1, h2, rev);

                            Clones ci1 = cloneInfoMap.get(gcid1);
                            Clones ci2 = cloneInfoMap.get(gcid2);
                            int similarity = ci1 != null ? ci1.similarity : 0;
                            int sameFile   = gcidToFilename.getOrDefault(gcid1, "")
                                                .equals(gcidToFilename.getOrDefault(gcid2, "")) ? 1 : 0;
                            int depth      = calculatePathDepth(
                                                gcidToFullPath.getOrDefault(gcid1, ""),
                                                gcidToFullPath.getOrDefault(gcid2, ""));

                            // New enrichment fields
                            String revAuthor = revAuthorMap.getOrDefault(rev, "UNKNOWN");
                            int sameAuthor   = 1; // single committer per revision in this VCS model
                            int classSize    = classToClonesMap.getOrDefault(classId, Collections.emptyList()).size();
                            int isSpcp       = (spcpPairs.contains(gcid1 + "-" + gcid2)
                                                || spcpPairs.contains(gcid2 + "-" + gcid1)) ? 1 : 0;
                            double wcsRecent = computeWeightedCouplingStrength(
                                                h1, h2, Math.max(pStart, rev - WCS_RECENT_WINDOW), rev);
                            int soloAfter    = countSoloAfterLastCoChange(h1, h2, rev);

                            // Fragment size & cumulative process metrics (up to current rev)
                            int nlines1  = ci1 != null ? ci1.nLines : 0;
                            int nlines2  = ci2 != null ? ci2.nLines : 0;
                            int noc1     = changesSoFar.getOrDefault(gcid1, 0);
                            int noc2     = changesSoFar.getOrDefault(gcid2, 0);
                            int fileAge1 = rev - h1.addedInRev;
                            int fileAge2 = rev - h2.addedInRev;
                            // Churn and ownership are full-lifetime values from computeProcessAndOwnershipMetrics
                            int churn1   = h1.churn;   int churn2   = h2.churn;
                            int da1      = h1.distinctAuthors;   int da2 = h2.distinctAuthors;
                            double map1  = h1.majorAuthorProportion; double map2 = h2.majorAuthorProportion;
                            int mac1     = h1.minorAuthorCount;  int mac2 = h2.minorAuthorCount;

                            // Decay-weighted features (5 half-lives × 3 metrics)
                            Set<Integer> spcpRevsUpToNow = new TreeSet<>();
                            for (int r : pairSPCPRevisions.getOrDefault(pairKey, Collections.emptySet()))
                                if (r <= rev) spcpRevsUpToNow.add(r);
                            Set<Integer> coChangeRevsUpToNow = new TreeSet<>(h1.changeRevs);
                            coChangeRevsUpToNow.retainAll(h2.changeRevs);
                            coChangeRevsUpToNow.removeIf(r -> r > rev);

                            StringBuilder pairRow = new StringBuilder();
                            pairRow.append(rev).append(",").append(gcid1).append(",").append(gcid2).append(",")
                                .append(classId).append(",").append(ct1).append(",").append(ct2).append(",")
                                .append(cum[0]).append(",").append(cum[1]).append(",").append(cum[2]).append(",")
                                .append(String.format("%.4f", wcs)).append(",")
                                .append(String.format("%.4f", trend)).append(",")
                                .append(lcca).append(",").append(lsca).append(",")
                                .append(similarity).append(",").append(sameFile).append(",").append(depth).append(",")
                                .append(sameAuthor).append(",").append(classSize).append(",").append(isSpcp).append(",")
                                .append(String.format("%.4f", wcsRecent)).append(",").append(soloAfter).append(",")
                                .append(nlines1).append(",").append(nlines2).append(",")
                                .append(noc1).append(",").append(noc2).append(",")
                                .append(fileAge1).append(",").append(fileAge2).append(",")
                                .append(churn1).append(",").append(churn2).append(",")
                                .append(da1).append(",").append(da2).append(",")
                                .append(String.format("%.4f", map1)).append(",").append(String.format("%.4f", map2)).append(",")
                                .append(mac1).append(",").append(mac2);
                            for (int hi = 0; hi < HALF_LIVES.length; hi++) {
                                double lambda = lambdaFromHalfLife(HALF_LIVES[hi]);
                                pairRow.append(",").append(String.format("%.4f",
                                    computeDecayWeightedIR(h1, h2, pStart, rev, lambda)));
                            }
                            for (int hi = 0; hi < HALF_LIVES.length; hi++) {
                                double lambda = lambdaFromHalfLife(HALF_LIVES[hi]);
                                pairRow.append(",").append(String.format("%.4f",
                                    computeDecayWeightedCS(h1, h2, pStart, rev, lambda)));
                            }
                            for (int hi = 0; hi < HALF_LIVES.length; hi++) {
                                double lambda = lambdaFromHalfLife(HALF_LIVES[hi]);
                                pairRow.append(",").append(String.format("%.4f",
                                    computeDecayWeightedSPCP(spcpRevsUpToNow, coChangeRevsUpToNow, rev, lambda)));
                            }
                            pairRow.append("\n");
                            pairBw.write(pairRow.toString());

                            // Accumulate for Layer 3
                            gcidWcsAccum.computeIfAbsent(gcid1, k -> new ArrayList<>()).add(wcs);
                            gcidWcsAccum.computeIfAbsent(gcid2, k -> new ArrayList<>()).add(wcs);
                        }
                    }
                }

                // ----- Layer 1 + Layer 3: fragment rows -----
                List<Integer> activeGcids = new ArrayList<>(revStatus.keySet());
                Collections.sort(activeGcids);

                for (int gcid : activeGcids) {
                    if (!interestingGcids.contains(gcid)) continue;
                    CloneHistory h = historyMap.get(gcid);
                    if (h == null) continue;
                    Clones c = cloneInfoMap.get(gcid);

                    String ct       = revStatus.get(gcid);
                    // Only emit rows where the clone actually changed this revision
                    if (!"M".equalsIgnoreCase(ct)) continue;
                    int tc          = changesSoFar.getOrDefault(gcid, 0);
                    int tu          = unchangedSoFar.getOrDefault(gcid, 0);
                    int lifespan    = rev - h.addedInRev;
                    int totalRevs   = lifespan + 1;
                    double stabIdx  = totalRevs > 0 ? (double) tu / totalRevs : 0.0;
                    double changePr = totalRevs > 0 ? (double) tc / totalRevs : 0.0;
                    String author   = revAuthorMap.getOrDefault(rev, "UNKNOWN");
                    int nlines      = c != null ? c.nLines : 0;

                    // Layer 3: aggregate WCS across all active pairs for this gcid
                    List<Double> wcsList = gcidWcsAccum.getOrDefault(gcid, Collections.emptyList());
                    int numPairs = wcsList.size();
                    double wcsSum = 0, wcsMax = 0, wcsMin = numPairs > 0 ? 1.0 : 0.0;
                    int stronglyCoupled = 0, decoupled = 0;
                    for (double w : wcsList) {
                        wcsSum += w;
                        if (w > wcsMax) wcsMax = w;
                        if (w < wcsMin) wcsMin = w;
                        if (w >= 0.5) stronglyCoupled++;
                        if (w < 0.2)  decoupled++;
                    }
                    double wcsMean = numPairs > 0 ? wcsSum / numPairs : 0.0;
                    if (numPairs == 0) wcsMin = 0.0;

                    fragBw.write(
                        rev + "," + gcid + "," +
                        quote(gcidToFilename.getOrDefault(gcid, "")) + "," +
                        ct + "," + nlines + "," +
                        tc + "," + tu + "," +
                        String.format("%.4f", stabIdx) + "," +
                        String.format("%.4f", changePr) + "," +
                        lifespan + "," +
                        quote(author) + "," +
                        numPairs + "," +
                        String.format("%.4f", wcsMean) + "," +
                        String.format("%.4f", wcsMax) + "," +
                        String.format("%.4f", wcsMin) + "," +
                        stronglyCoupled + "," + decoupled + "\n");
                }

                System.out.println("Rev " + rev + ": " + revStatus.size() + " active clones written");
            }
        }

        System.out.println("=== Revision-Wise Export Complete ===");
        System.out.println("Fragment CSV: " + fragCsvPath);
        System.out.println("Pair CSV: " + pairCsvPath);
    }

    // ============== PROCESS & OWNERSHIP METRIC HELPERS ==============

    /**
     * Parse a ChangeInfo revision file and compute lines-changed per source file.
     *
     * File format (pipe-delimited):
     *   lineNum | prevRev | prevFile | prevLineRange | changeType(c/a/d) | currRev | currFile | currLineRange | fixType
     *
     * We sum the line-span of each currLineRange hunk to get total churn per file.
     *
     * @param rev  revision number whose ChangeInfo file to read
     * @return Map from bare filename (e.g. "asp.c") to total lines changed
     */
    private Map<String, Integer> buildChurnMap(int rev) {
        Map<String, Integer> churnByFile = new HashMap<>();
        String path = cp.getChangeRevisionInfoPath(rev);
        File f = new File(path);
        if (!f.exists()) {
            return churnByFile; // no ChangeInfo for this revision — churn stays 0
        }
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                // Split on " | " (space-pipe-space)
                String[] parts = line.split("\\s*\\|\\s*");
                // Expected minimum: lineNum | prevRev | prevFile | prevRange | type | currRev | currFile | currRange
                if (parts.length < 8) continue;
                String currFile = parts[6].trim(); // e.g. "Revision_461/sql.c"
                String currRange = parts[7].trim(); // e.g. "957,1051" or "290"
                // Extract bare filename
                String bareName = extractFileName(currFile);
                // Parse line span: "start,end" or just "start"
                int lines = 1;
                if (currRange.contains(",")) {
                    String[] rp = currRange.split(",");
                    try {
                        int lo = Integer.parseInt(rp[0].trim());
                        int hi = Integer.parseInt(rp[1].trim());
                        lines = Math.abs(hi - lo) + 1;
                    } catch (NumberFormatException ignored) { }
                }
                churnByFile.merge(bareName, lines, Integer::sum);
            }
        } catch (IOException e) {
            System.out.println("Warning: could not read ChangeInfo for rev " + rev + ": " + e.getMessage());
        }
        return churnByFile;
    }

    /**
     * Compute process and ownership metrics for every CloneHistory entry and
     * store the results directly in the CloneHistory fields.
     *
     * Metrics computed:
     *   - noc               = changeRevs.size()
     *   - fileAge           = endRev - addedInRev
     *   - churn             = sum of lines changed in the clone's file across all changeRevs
     *   - distinctAuthors   = |{unique authors at changeRevs}|
     *   - majorAuthorProportion = (commits by top author) / noc
     *   - minorAuthorCount  = authors contributing < 5% of commits
     *
     * @param historyMap     gcid → CloneHistory
     * @param revAuthorMap   revision → author string
     * @param gcidToFilename gcid → bare filename (e.g. "asp.c")
     */
    private void computeProcessAndOwnershipMetrics(
            Map<Integer, CloneHistory> historyMap,
            Map<Integer, String> revAuthorMap,
            Map<Integer, String> gcidToFilename) {

        // Cache churn maps to avoid re-reading the same file for multiple clones
        Map<Integer, Map<String, Integer>> churnCache = new HashMap<>();

        for (Map.Entry<Integer, CloneHistory> entry : historyMap.entrySet()) {
            int gcid = entry.getKey();
            CloneHistory h = entry.getValue();

            // --- Process metrics ---
            h.noc     = h.changeRevs.size();
            h.fileAge = h.endRev - h.addedInRev;

            // Churn: sum lines changed in this clone's file across its changeRevs
            String fname = gcidToFilename.getOrDefault(gcid, "");
            int totalChurn = 0;
            for (int rev : h.changeRevs) {
                Map<String, Integer> cm = churnCache.computeIfAbsent(rev, this::buildChurnMap);
                totalChurn += cm.getOrDefault(fname, 0);
            }
            h.churn = totalChurn;

            // --- Ownership metrics ---
            // Collect author frequency across changeRevs
            Map<String, Integer> authorCount = new HashMap<>();
            for (int rev : h.changeRevs) {
                String author = revAuthorMap.getOrDefault(rev, "UNKNOWN");
                authorCount.merge(author, 1, Integer::sum);
            }

            h.distinctAuthors = authorCount.size();

            if (h.noc == 0) {
                h.majorAuthorProportion = 0.0;
                h.minorAuthorCount      = 0;
            } else {
                int maxContrib = authorCount.values().stream().mapToInt(Integer::intValue).max().orElse(0);
                h.majorAuthorProportion = (double) maxContrib / h.noc;

                double threshold = 0.05 * h.noc; // < 5 % of commits
                h.minorAuthorCount = (int) authorCount.values().stream()
                        .filter(c -> c < threshold)
                        .count();
            }
        }

        System.out.println("Process + ownership metrics computed for " + historyMap.size() + " clones");
    }

    /**
     * Load method info (method_name, method_signature) from GlobalCloneInfo files.
     * Scans all revision files and builds a mapping from globalCloneId to method
     * info.
     */
    private Map<Integer, CloneMethodInfo> loadMethodInfoFromGlobalCloneInfo() {
        Map<Integer, CloneMethodInfo> result = new HashMap<>();

        System.out.println("Loading method info from GlobalCloneInfo files...");

        for (int rev = startRev; rev <= endRev; rev++) {
            String path = cp.WorkFolder + cp.getSubjectSystem() + "/Clones/" + cChoice + "/" +
                    granularity + "/GlobalCloneInfo/" + rev + "_clones.txt";

            File file = new File(path);
            if (!file.exists()) {
                continue;
            }

            try (BufferedReader br = new BufferedReader(new FileReader(path))) {
                String header = br.readLine(); // Skip header
                if (header == null)
                    continue;

                // Detect column indices from header
                // Expected format: globalCloneId | filePath | StartLine | EndLine | classId |
                // globalMethodId | method_name | method_signature | methodStart | methodEnd
                String[] headerParts = header.split("\\|");
                int methodNameIdx = -1;
                int methodSignatureIdx = -1;
                for (int i = 0; i < headerParts.length; i++) {
                    String col = headerParts[i].trim().toLowerCase();
                    if (col.equals("method_name"))
                        methodNameIdx = i;
                    else if (col.equals("method_signature"))
                        methodSignatureIdx = i;
                }

                String line;
                while ((line = br.readLine()) != null) {
                    if (line.trim().isEmpty())
                        continue;

                    String[] parts = line.split("\\|");
                    if (parts.length < 7)
                        continue;

                    try {
                        int gcid = Integer.parseInt(parts[0].trim());

                        // Only add if not already present (use first occurrence)
                        if (!result.containsKey(gcid)) {
                            CloneMethodInfo info = new CloneMethodInfo();

                            // Get method_name
                            if (methodNameIdx >= 0 && methodNameIdx < parts.length) {
                                String methodName = parts[methodNameIdx].trim();
                                info.methodName = methodName.equals("NULL") ? null : methodName;
                            }

                            // Get method_signature
                            if (methodSignatureIdx >= 0 && methodSignatureIdx < parts.length) {
                                String methodSig = parts[methodSignatureIdx].trim();
                                info.methodSignature = methodSig.equals("NULL") ? null : methodSig;
                            }

                            result.put(gcid, info);
                        }
                    } catch (NumberFormatException e) {
                        // Skip malformed lines
                    }
                }
            } catch (IOException e) {
                System.out.println("Error reading GlobalCloneInfo file: " + path);
            }
        }

        System.out.println("Loaded method info for " + result.size() + " clones");
        return result;
    }

    /**
     * Compute exponential time-decay weighted coupling strength (WCS) for a clone pair.
     *
     * WCS = Σ W(co-change events up to maxRev) / Σ W(all change events up to maxRev)
     * where W(event) = e^(-DECAY_LAMBDA * (maxRev - eventRevision))
     *
     * Returns 0.0 if there are no changes in the window.
     *
     * @param h1     CloneHistory of the first clone
     * @param h2     CloneHistory of the second clone
     * @param maxRev upper bound of revision window (typically current or last revision)
     * @param minRev lower bound of revision window (use 0 to include everything)
     */
    private double computeWeightedCouplingStrength(CloneHistory h1, CloneHistory h2,
            int minRev, int maxRev) {
        double coChangeWeight = 0.0;
        double totalWeight = 0.0;

        // Collect all revisions where either clone changed
        Set<Integer> allChangeRevs = new HashSet<>();
        allChangeRevs.addAll(h1.changeRevs);
        allChangeRevs.addAll(h2.changeRevs);

        for (int rev : allChangeRevs) {
            if (rev < minRev || rev > maxRev) continue;

            int age = maxRev - rev;
            double w = Math.exp(-DECAY_LAMBDA * age);

            boolean c1Changed = h1.changeRevs.contains(rev);
            boolean c2Changed = h2.changeRevs.contains(rev);

            if (c1Changed && c2Changed) {
                coChangeWeight += w; // co-change: counts toward numerator
            }
            totalWeight += w; // any change counts toward denominator
        }

        return totalWeight == 0.0 ? 0.0 : coChangeWeight / totalWeight;
    }

    /**
     * Compute coupling trend: WCS in the first half of pair lifetime vs second half.
     * Negative value means coupling is weakening (decoupling trend).
     */
    private double computeCouplingTrend(CloneHistory h1, CloneHistory h2,
            int pairStartRev, int pairEndRev) {
        if (pairEndRev <= pairStartRev) return 0.0;
        int midRev = pairStartRev + (pairEndRev - pairStartRev) / 2;
        double earlyWCS = computeWeightedCouplingStrength(h1, h2, pairStartRev, midRev);
        double recentWCS = computeWeightedCouplingStrength(h1, h2, midRev + 1, pairEndRev);
        return recentWCS - earlyWCS;
    }

    /**
     * Returns the age (in revisions) of the last co-change event up to maxRev,
     * or -1 if there was no co-change.
     */
    private int lastCoChangeAge(CloneHistory h1, CloneHistory h2, int maxRev) {
        int lastCoRev = -1;
        for (int rev : h1.changeRevs) {
            if (rev <= maxRev && h2.changeRevs.contains(rev)) {
                lastCoRev = Math.max(lastCoRev, rev);
            }
        }
        return lastCoRev == -1 ? -1 : maxRev - lastCoRev;
    }

    /**
     * Returns the age of the most recent solo change (either clone changed alone) up to maxRev,
     * or -1 if no solo change ever occurred.
     */
    private int lastSoloChangeAge(CloneHistory h1, CloneHistory h2, int maxRev) {
        int lastSoloRev = -1;
        for (int rev : h1.changeRevs) {
            if (rev <= maxRev && !h2.changeRevs.contains(rev)) {
                lastSoloRev = Math.max(lastSoloRev, rev);
            }
        }
        for (int rev : h2.changeRevs) {
            if (rev <= maxRev && !h1.changeRevs.contains(rev)) {
                lastSoloRev = Math.max(lastSoloRev, rev);
            }
        }
        return lastSoloRev == -1 ? -1 : maxRev - lastSoloRev;
    }

    /**
     * Count solo change events (either clone changed alone) that occurred strictly
     * after the last co-change up to maxRev. If there was never a co-change, counts
     * all solo events up to maxRev.
     */
    private int countSoloAfterLastCoChange(CloneHistory h1, CloneHistory h2, int maxRev) {
        int lastCoRev = -1;
        for (int rev : h1.changeRevs) {
            if (rev <= maxRev && h2.changeRevs.contains(rev)) {
                lastCoRev = Math.max(lastCoRev, rev);
            }
        }
        int count = 0;
        for (int rev : h1.changeRevs) {
            if (rev <= maxRev && rev > lastCoRev && !h2.changeRevs.contains(rev)) count++;
        }
        for (int rev : h2.changeRevs) {
            if (rev <= maxRev && rev > lastCoRev && !h1.changeRevs.contains(rev)) count++;
        }
        return count;
    }

    // ============== DECAY-WEIGHTED METRIC METHODS ==============

    /**
     * Decay-weighted Independence Ratio (IR_decay).
     *
     * IR_decay = Σ w(indep_changes) / (Σ w(co_changes) + Σ w(indep_changes) + ε)
     *
     * where w(event at revision r) = e^(-λ * (maxRev - r))
     *
     * Higher IR_decay → pair is trending toward independent evolution.
     * The advisor's example: co-changes at 15,30,45 and independent at 60,75,90
     * with λ=0.03, r_current=100 gives IR_decay ≈ 0.79 (vs unweighted 0.50).
     */
    private double computeDecayWeightedIR(CloneHistory h1, CloneHistory h2,
            int minRev, int maxRev, double lambda) {
        double coWeight = 0.0;
        double indepWeight = 0.0;

        // Collect all revisions where either clone changed
        Set<Integer> allChangeRevs = new HashSet<>();
        allChangeRevs.addAll(h1.changeRevs);
        allChangeRevs.addAll(h2.changeRevs);

        for (int rev : allChangeRevs) {
            if (rev < minRev || rev > maxRev) continue;

            double w = Math.exp(-lambda * (maxRev - rev));
            boolean c1Changed = h1.changeRevs.contains(rev);
            boolean c2Changed = h2.changeRevs.contains(rev);

            if (c1Changed && c2Changed) {
                coWeight += w;
            } else {
                // One changed, the other didn't → independent change
                indepWeight += w;
            }
        }

        double denom = coWeight + indepWeight + 1e-9;
        return indepWeight / denom;
    }

    /**
     * Decay-weighted Coupling Strength (CS_decay) — advisor's max-denominator formulation.
     *
     * CS_decay = Σ w(co_changes) / max(Σ w(gc1_all_changes), Σ w(gc2_all_changes)) + ε
     *
     * This differs from the existing WCS (union denominator / Jaccard-style).
     * CS_decay captures "what fraction of the more-active clone's changes are co-changes",
     * making it sensitive to asymmetric pairs where one clone changes much more than the other.
     */
    private double computeDecayWeightedCS(CloneHistory h1, CloneHistory h2,
            int minRev, int maxRev, double lambda) {
        double coWeight = 0.0;
        double gc1Weight = 0.0;
        double gc2Weight = 0.0;

        // Weighted sum of gc1's changes
        for (int rev : h1.changeRevs) {
            if (rev < minRev || rev > maxRev) continue;
            double w = Math.exp(-lambda * (maxRev - rev));
            gc1Weight += w;
            if (h2.changeRevs.contains(rev)) {
                coWeight += w;
            }
        }

        // Weighted sum of gc2's changes (only for denominator — co-change already counted)
        for (int rev : h2.changeRevs) {
            if (rev < minRev || rev > maxRev) continue;
            double w = Math.exp(-lambda * (maxRev - rev));
            gc2Weight += w;
        }

        double maxWeight = Math.max(gc1Weight, gc2Weight);
        return maxWeight < 1e-9 ? 0.0 : coWeight / (maxWeight + 1e-9);
    }

    /**
     * Dynamically compute SPCP status for each co-change revision of a pair.
     * For each co-change, fetches source fragments and checks if similarity
     * is preserved after the change (using LCS-based token similarity).
     *
     * @param gcid1  global clone id of first clone
     * @param gcid2  global clone id of second clone
     * @param h1     CloneHistory of first clone
     * @param h2     CloneHistory of second clone
     * @param maxRev upper bound revision
     * @param cloneInstancesByRev  gcid → (revision → Clones) mapping for fragment lookup
     * @return set of revision numbers where the co-change was similarity-preserving
     */
    private Set<Integer> computeSPCPRevisions(int gcid1, int gcid2,
            CloneHistory h1, CloneHistory h2, int maxRev,
            Map<Integer, Map<Integer, Clones>> cloneInstancesByRev) {

        Set<Integer> coChangeRevs = new TreeSet<>(h1.changeRevs);
        coChangeRevs.retainAll(h2.changeRevs);

        Set<Integer> spcpRevs = new TreeSet<>();

        for (int rev : coChangeRevs) {
            if (rev > maxRev) continue;

            // Find next revision where BOTH clones exist (to compare after-state)
            int nextRev = findNextRevWithBothClones(gcid1, gcid2, rev, cloneInstancesByRev);
            if (nextRev == -1) continue;

            // Fetch fragments
            String before1 = getFragmentText(gcid1, rev, cloneInstancesByRev);
            String before2 = getFragmentText(gcid2, rev, cloneInstancesByRev);
            String after1  = getFragmentText(gcid1, nextRev, cloneInstancesByRev);
            String after2  = getFragmentText(gcid2, nextRev, cloneInstancesByRev);

            if (before1 == null || before2 == null || after1 == null || after2 == null) continue;

            double simAfter = lcsSimilarity(after1, after2);
            if (simAfter >= spcpSimThreshold) {
                spcpRevs.add(rev);
            }
        }

        return spcpRevs;
    }

    /**
     * Decay-weighted SPCP fraction.
     *
     * SPCP_decay = Σ w(SPCP-verified co-change revisions) / (Σ w(all co-change revisions) + ε)
     *
     * Captures whether the pair is *currently* following similarity-preserving patterns,
     * not whether it did so historically. A pair that was SPCP for 80 revisions but
     * broke the pattern in the last 10 gets a low SPCP_decay.
     */
    private double computeDecayWeightedSPCP(Set<Integer> spcpRevs,
            Set<Integer> allCoChangeRevs, int maxRev, double lambda) {
        double spcpWeight = 0.0;
        double totalWeight = 0.0;

        for (int rev : allCoChangeRevs) {
            if (rev > maxRev) continue;
            double w = Math.exp(-lambda * (maxRev - rev));
            totalWeight += w;
            if (spcpRevs.contains(rev)) {
                spcpWeight += w;
            }
        }

        return totalWeight < 1e-9 ? 0.0 : spcpWeight / totalWeight;
    }

    // ============== FRAGMENT FETCHING FOR DYNAMIC SPCP ==============

    /**
     * Find next revision after currentRev where BOTH clones have instance data.
     */
    private int findNextRevWithBothClones(int gcid1, int gcid2, int currentRev,
            Map<Integer, Map<Integer, Clones>> cloneInstancesByRev) {
        Map<Integer, Clones> instances1 = cloneInstancesByRev.get(gcid1);
        Map<Integer, Clones> instances2 = cloneInstancesByRev.get(gcid2);
        if (instances1 == null || instances2 == null) return -1;

        // Find sorted revisions after currentRev where both exist
        TreeSet<Integer> revs1 = new TreeSet<>(instances1.keySet());
        for (int nextRev : revs1.tailSet(currentRev + 1)) {
            if (instances2.containsKey(nextRev)) {
                return nextRev;
            }
        }
        return -1;
    }

    /**
     * Get the source code text of a clone fragment at a specific revision.
     * Uses the clone's filePath, startLine, endLine to read from the source files.
     */
    private String getFragmentText(int gcid, int revision,
            Map<Integer, Map<Integer, Clones>> cloneInstancesByRev) {
        Map<Integer, Clones> instances = cloneInstancesByRev.get(gcid);
        if (instances == null) return null;
        Clones c = instances.get(revision);
        if (c == null) return null;

        // Build the full path to the source file
        // filePath already contains "Revision_N/..." prefix
        Path filePath = Paths.get(cp.getCloneDir(), c.filePath);
        if (!Files.exists(filePath)) return null;

        try {
            // Use ISO-8859-1 to handle non-UTF-8 characters in C source files
            List<String> lines = Files.readAllLines(filePath, java.nio.charset.StandardCharsets.ISO_8859_1);

            int start = Math.max(0, c.startLine - 1);
            int end = Math.min(lines.size() - 1, c.endLine - 1);
            if (start > end || start >= lines.size()) return null;

            StringBuilder sb = new StringBuilder();
            for (int i = start; i <= end; i++) {
                sb.append(lines.get(i)).append("\n");
            }
            return sb.toString();
        } catch (IOException e) {
            return null;
        }
    }

    // ============== LCS SIMILARITY (ported from SPCPAnalysis) ==============

    private static List<String> tokenize(String s) {
        if (s == null) return Collections.emptyList();
        s = s.replaceAll("[^A-Za-z0-9_]", " ");
        String[] toks = s.trim().split("\\s+");
        List<String> out = new ArrayList<>();
        for (String t : toks)
            if (!t.isEmpty()) out.add(t);
        return out;
    }

    private static int lcsLength(List<String> a, List<String> b) {
        int n = a.size(), m = b.size();
        if (n == 0 || m == 0) return 0;
        int[] prev = new int[m + 1], cur = new int[m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++)
                cur[j] = a.get(i - 1).equals(b.get(j - 1)) ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
            int[] tmp = prev;
            prev = cur;
            cur = tmp;
            java.util.Arrays.fill(cur, 0);
        }
        return prev[m];
    }

    private static double lcsSimilarity(String a, String b) {
        List<String> ta = tokenize(a), tb = tokenize(b);
        if (ta.isEmpty() || tb.isEmpty()) return 0.0;
        return (double) lcsLength(ta, tb) / Math.max(ta.size(), tb.size());
    }


    /**
     * Quote a string for CSV output (escape existing quotes)
     */
    private String quote(String s) {
        if (s == null)
            return "";
        if (s.contains(",") || s.contains("\"") || s.contains("\n")) {
            return "\"" + s.replace("\"", "\"\"") + "\"";
        }
        return s;
    }

    /**
     * Calculate the folder depth/distance between two file paths.
     * - Same file: depth = 0
     * - Same folder, different file: depth = 1
     * - Different folders: count the number of folder hops needed
     * 
     * Example:
     * "src/main/foo.c" vs "src/main/bar.c" = 1 (different file, same folder)
     * "src/main/foo.c" vs "src/util/bar.c" = 2 (up 1, down 1)
     * "src/a/b/foo.c" vs "src/x/y/bar.c" = 4 (up 2, down 2)
     */
    private int calculatePathDepth(String path1, String path2) {
        if (path1 == null || path2 == null) {
            return -1; // Unknown
        }

        // Normalize paths: replace backslashes with forward slashes
        path1 = path1.replace("\\", "/");
        path2 = path2.replace("\\", "/");

        // Remove any Revision_* prefixes for clean comparison
        path1 = path1.replaceAll("Revision_\\d+/", "");
        path2 = path2.replaceAll("Revision_\\d+/", "");

        // Same file = depth 0
        if (path1.equals(path2)) {
            return 0;
        }

        // Split into path components
        String[] parts1 = path1.split("/");
        String[] parts2 = path2.split("/");

        // Get directory parts (all but last component which is the filename)
        String[] dir1 = new String[Math.max(0, parts1.length - 1)];
        String[] dir2 = new String[Math.max(0, parts2.length - 1)];
        System.arraycopy(parts1, 0, dir1, 0, dir1.length);
        System.arraycopy(parts2, 0, dir2, 0, dir2.length);

        // Find common prefix length
        int commonPrefixLen = 0;
        int minLen = Math.min(dir1.length, dir2.length);
        for (int i = 0; i < minLen; i++) {
            if (dir1[i].equals(dir2[i])) {
                commonPrefixLen++;
            } else {
                break;
            }
        }

        // Depth = steps up from path1 to common ancestor + steps down to path2 + 1 for
        // different file
        int stepsUp = dir1.length - commonPrefixLen;
        int stepsDown = dir2.length - commonPrefixLen;
        int depth = stepsUp + stepsDown;

        // If directories are the same, depth is 1 (just different files)
        // If directories differ, add the folder traversal
        return depth == 0 ? 1 : depth;
    }

    /**
     * Load SPCP pairs from the SPCP analysis CSV file.
     * Returns a set of "gcid1-gcid2" strings for pairs that have
     * no_of_revision_paired > 0.
     */
    private Set<String> loadSPCPPairs() {
        Set<String> spcpPairs = new HashSet<>();

        // Build the SPCP file path:
        // WorkFolder/{SubjectSystem}/SPCP_Type{N}_{Granularity}.csv
        String spcpPath = "WorkFolder/" + cp.getSubjectSystem() + "/SPCP_" + cChoice + "_" + granularity + ".csv";
        File spcpFile = new File(spcpPath);

        if (!spcpFile.exists()) {
            System.out.println("SPCP file not found: " + spcpPath + " - is_spcp will be false for all pairs");
            return spcpPairs;
        }

        System.out.println("Loading SPCP pairs from: " + spcpPath);

        try (BufferedReader br = new BufferedReader(new FileReader(spcpFile))) {
            String header = br.readLine(); // Skip header
            if (header == null) {
                return spcpPairs;
            }

            // Find column indices from header
            String[] headerParts = header.split(",");
            int gcid1Idx = -1, gcid2Idx = -1, revPairedIdx = -1;
            for (int i = 0; i < headerParts.length; i++) {
                String col = headerParts[i].trim().toLowerCase();
                if (col.equals("gcid1"))
                    gcid1Idx = i;
                else if (col.equals("gcid2"))
                    gcid2Idx = i;
                else if (col.equals("no_of_revision_paired"))
                    revPairedIdx = i;
            }

            if (gcid1Idx == -1 || gcid2Idx == -1 || revPairedIdx == -1) {
                System.out.println("Warning: SPCP file missing required columns (gcid1, gcid2, no_of_revision_paired)");
                return spcpPairs;
            }

            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length > Math.max(gcid1Idx, Math.max(gcid2Idx, revPairedIdx))) {
                    try {
                        int gcid1 = Integer.parseInt(parts[gcid1Idx].trim());
                        int gcid2 = Integer.parseInt(parts[gcid2Idx].trim());
                        int revPaired = Integer.parseInt(parts[revPairedIdx].trim());

                        if (revPaired > 0) {
                            // Store both orderings for easy lookup
                            spcpPairs.add(gcid1 + "-" + gcid2);
                        }
                    } catch (NumberFormatException e) {
                        // Skip malformed lines
                    }
                }
            }
        } catch (IOException e) {
            System.out.println("Error reading SPCP file: " + e.getMessage());
        }

        System.out.println("Loaded " + spcpPairs.size() + " SPCP pairs");
        return spcpPairs;
    }
}
