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

                        pairs.add(pair);
                    }
                }
            }
        }

        // Save pairs to CSV
        String pairOutputPath = cp.getGenealogyPath(cChoice, granularity).replace(".csv", "_pairs.csv");
        BufferedWriter bw = new BufferedWriter(new FileWriter(pairOutputPath));

        bw.write(
                "pairId,cloneType,granularity,classId,clone1_gcid,clone2_gcid,clone1_addedRev,clone2_addedRev," +
                        "clone1_deletedRev,clone2_deletedRev,clone1_lifespan,clone2_lifespan," +
                        "co_change,gcid1_independent_change,gcid2_independent_change,similarity," +
                        "filepath1,filepath2,file_distance,gcid1_authors,gcid2_authors\n");

        for (ClonePair pair : pairs) {
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
                            quote(pair.gcid2Authors) +
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
        // New fields for enhanced analysis
        int coChange;
        int gcid1IndependentChange;
        int gcid2IndependentChange;
        int similarity;
        String filepath1;
        String filepath2;
        int fileDistance;
        String gcid1Authors;
        String gcid2Authors;
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
            }
            revisionCloneStatus.put(rev, revStatus);
        }

        System.out.println("Total unique clones found: " + cloneInfoMap.size());

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

                    // Track cumulative counts
                    int coChangeCount = 0;
                    int gcid1IndependentCount = 0;
                    int gcid2IndependentCount = 0;

                    // Analyze each revision where at least one clone exists
                    int pairStartRev = Math.max(h1.addedInRev, h2.addedInRev);
                    int pairEndRev = Math.min(h1.endRev, h2.endRev);

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
            // Write header
            bw.write("Revision,gcid1,gcid2,classid,co_change_count,gcid1_independent_change_count," +
                    "gcid2_independent_change_count,filepath1,filepath2,author1,author2,similarity," +
                    "clone_type,nlines1,nlines2,depth,class_size,is_spcp," +
                    "method_name1,method_signature1,method_name2,method_signature2\n");

            // Write records
            for (EvolutionRecord rec : records) {
                bw.write(String.format("%d,%d,%d,%d,%d,%d,%d,%s,%s,%s,%s,%d,%s,%d,%d,%d,%d,%d,%s,%s,%s,%s\n",
                        rec.revision,
                        rec.gcid1,
                        rec.gcid2,
                        rec.classId,
                        rec.coChangeCount,
                        rec.gcid1IndependentCount,
                        rec.gcid2IndependentCount,
                        quote(rec.filepath1),
                        quote(rec.filepath2),
                        quote(rec.author1),
                        quote(rec.author2),
                        rec.similarity,
                        rec.cloneType,
                        rec.nlines1,
                        rec.nlines2,
                        rec.depth,
                        rec.classSize,
                        rec.isSPCP ? 1 : 0,
                        quote(rec.methodName1),
                        quote(rec.methodSignature1),
                        quote(rec.methodName2),
                        quote(rec.methodSignature2)));
            }
        }

        System.out.println("=== Evolution Dataset Export Complete ===");
        System.out.println("Output file: " + outputPath);
        System.out.println("Total records: " + records.size());
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
