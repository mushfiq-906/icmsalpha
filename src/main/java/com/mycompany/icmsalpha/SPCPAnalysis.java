package com.mycompany.icmsalpha;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class SPCPAnalysis {

    // SPCP thresholds loaded from clone_config.properties
    private final double SIM_THRESHOLD;
    private final double MAX_DROP;
    private final double RESYNC_THRESHOLD;
    private final int MAX_LAG_REVISIONS;

    enum ChangeCategory {
        SIMILARITY_PRESERVING,
        RESYNCHRONIZING,
        DIVERGING,
        NONE
    }

    private final String cloneType;
    private String granularity;
    private final String systemPath;
    private final int startRev;
    private final int endRev;

    private final Map<Integer, Clones> cloneInfoMap = new HashMap<>();
    private final Map<Integer, CloneHistory> historyMap = new HashMap<>();
    private final Map<Integer, Map<Integer, Clones>> cloneInstancesByRevision = new HashMap<>();

    // Track which revisions each clone appears in
    private final Map<Integer, Set<Integer>> cloneRevisionMap = new HashMap<>();

    // Track available revisions (for finding next revision properly)
    private final Set<Integer> availableRevisions = new TreeSet<>();

    CommonParameters cp = new CommonParameters();
    LoadParam lp = new LoadParam();

    public SPCPAnalysis(String cloneType, String granularity, int startRev, int endRev) {
        // Load thresholds from config file
        this.SIM_THRESHOLD = lp.getSpcpSimilarity();
        this.MAX_DROP = lp.getSpcpMaxDrop();
        this.RESYNC_THRESHOLD = lp.getSpcpResync();
        this.MAX_LAG_REVISIONS = lp.getSpcpMaxLagRevisions();

        print("SPCP Thresholds loaded from config:");
        print("  SIM_THRESHOLD: " + SIM_THRESHOLD);
        print("  MAX_DROP: " + MAX_DROP);
        print("  RESYNC_THRESHOLD: " + RESYNC_THRESHOLD);
        print("  MAX_LAG_REVISIONS: " + MAX_LAG_REVISIONS);

        this.systemPath = cp.WorkFolder + cp.getSubjectSystem();
        this.cloneType = cloneType;
        if (granularity.equals("f")) {
            this.granularity = "Function";
        } else if (granularity.equals("b")) {
            this.granularity = "Block";
        }
        this.startRev = startRev;
        this.endRev = endRev;
        loadData();
    }

    public void print(Object o) {
        System.out.println(o);
    }

    // ============ FIXED: Load data from your actual file format ============
    private void loadData() {
        try {
            // FIX 1: Use proper file separator
            String basePath = systemPath + File.separator + "Clones" + File.separator +
                    cloneType + File.separator + granularity + File.separator + "GlobalClones";

            print("Loading from: " + basePath);
            File folder = new File(basePath);

            if (!folder.exists()) {
                print("ERROR: Folder does not exist: " + basePath);
                return;
            }

            File[] files = folder.listFiles();
            if (files == null || files.length == 0) {
                print("ERROR: No files found in: " + basePath);
                return;
            }

            print("Found " + files.length + " files in directory");

            // Load data from each revision's clone file
            for (int rev = startRev; rev <= endRev; rev++) {
                String cloneFile = basePath + File.separator + rev + "_clones.txt";
                File f = new File(cloneFile);

                if (!f.exists()) {
                    // Skip silently for revisions without clone files
                    continue;
                }

                // Track that this revision has data
                availableRevisions.add(rev);
                print("Loading: " + cloneFile);
                loadCloneFile(f, rev);
            }

            print("Total clones loaded: " + cloneInfoMap.size());
            print("Total clone classes: " + getClassCount());
            print("Available revisions: " + availableRevisions.size() + " (from " +
                    (availableRevisions.isEmpty() ? "N/A" : Collections.min(availableRevisions)) +
                    " to " + (availableRevisions.isEmpty() ? "N/A" : Collections.max(availableRevisions)) + ")");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // FIX 3: Parse your actual file format
    private void loadCloneFile(File file, int revision) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;

        // Skip header if exists
        line = br.readLine();
        if (line != null && line.startsWith("Revision")) {
            // This is header, skip it
        } else {
            // No header, process this line
            processCloneLine(line, revision);
        }

        // Process remaining lines
        while ((line = br.readLine()) != null) {
            processCloneLine(line, revision);
        }
        br.close();
    }

    // FIX 4: Parse according to your format:
    // Revision | filePath | changeType | StartLine | EndLine | Nlines | cloneId |
    // classId | globalCloneId
    private void processCloneLine(String line, int revision) {
        if (line == null || line.trim().isEmpty())
            return;

        String[] parts = line.split("\\|");
        if (parts.length < 9) {
            print("WARNING: Invalid line format: " + line);
            return;
        }

        try {
            // Parse fields (trim whitespace)
            int rev = Integer.parseInt(parts[0].trim());
            String filePath = parts[1].trim();
            String changeType = parts[2].trim();
            int startLine = Integer.parseInt(parts[3].trim());
            int endLine = Integer.parseInt(parts[4].trim());
            int nlines = Integer.parseInt(parts[5].trim());
            int cloneId = Integer.parseInt(parts[6].trim());
            int classId = Integer.parseInt(parts[7].trim());
            int globalCloneId = Integer.parseInt(parts[8].trim());

            // Create or update clone info
            if (!cloneInfoMap.containsKey(globalCloneId)) {
                Set<Integer> changeRevs = new TreeSet<>();
                // IMPORTANT: For SPCP, only track MODIFICATIONS (M/c), not additions (A/a)
                boolean isModification = changeType.equalsIgnoreCase("M") || changeType.equalsIgnoreCase("c");
                if (isModification) {
                    changeRevs.add(revision);
                }

                Clones clone = new Clones(globalCloneId, classId, filePath,
                        startLine, endLine, revision, changeRevs);
                cloneInfoMap.put(globalCloneId, clone);
                historyMap.put(globalCloneId, new CloneHistory(changeRevs, revision));
            } else {
                // Update existing clone with new revision info
                Clones existingClone = cloneInfoMap.get(globalCloneId);
                boolean isModification = changeType.equalsIgnoreCase("M") || changeType.equalsIgnoreCase("c");
                if (isModification) {
                    existingClone.changeRevs.add(revision);
                    historyMap.get(globalCloneId).changeRevs.add(revision);
                }
            }

            // Track which revisions this clone exists in
            cloneRevisionMap.computeIfAbsent(globalCloneId, k -> new TreeSet<>()).add(revision);

            // Store instance for this specific revision
            cloneInstancesByRevision.computeIfAbsent(globalCloneId, k -> new HashMap<>())
                    .put(revision, new Clones(globalCloneId, classId, filePath,
                            startLine, endLine, revision, new TreeSet<>()));

        } catch (NumberFormatException e) {
            print("ERROR parsing line: " + line);
            e.printStackTrace();
        }
    }

    private int getClassCount() {
        Set<Integer> uniqueClasses = new TreeSet<>();
        for (Clones c : cloneInfoMap.values()) {
            uniqueClasses.add(c.classId);
        }
        return uniqueClasses.size();
    }

    // ============ FIXED: Fragment fetcher with better error handling ============
    private interface FragmentFetcher {
        String getFragmentTextAt(int globalCloneId, int revision) throws IOException;
    }

    private class FileBasedFragmentFetcher implements FragmentFetcher {
        @Override
        public String getFragmentTextAt(int globalCloneId, int revision) throws IOException {
            Map<Integer, Clones> revMap = cloneInstancesByRevision.get(globalCloneId);
            if (revMap == null) {
                print("    Fragment fetch: No revision map for gcid=" + globalCloneId);
                return null;
            }

            Clones c = revMap.get(revision);
            if (c == null) {
                print("    Fragment fetch: Clone not found in revision " + revision);
                return null;
            }

            // FIX 5: Construct proper path to revision files
            String revisionFolder = "Revision_" + revision;
            Path filePath = Paths.get(cp.getCloneDir(), c.filePath);

            if (!Files.exists(filePath)) {
                print("    Fragment fetch: File does not exist: " + filePath);
                return null;
            }

            // FIX: Use ISO-8859-1 charset to handle non-UTF-8 characters in C source files
            List<String> lines;
            try {
                lines = Files.readAllLines(filePath, java.nio.charset.StandardCharsets.ISO_8859_1);
            } catch (IOException e) {
                print("    Fragment fetch: Error reading file: " + e.getMessage());
                return null;
            }

            // FIX 6: Handle line indices correctly (1-based to 0-based)
            int start = Math.max(0, c.startLine - 1);
            int end = Math.min(lines.size() - 1, c.endLine - 1);

            if (start > end || start >= lines.size()) {
                print("    Fragment fetch: Invalid line range [" + c.startLine + "," + c.endLine + "]");
                return null;
            }

            StringBuilder sb = new StringBuilder();
            for (int i = start; i <= end; i++) {
                sb.append(lines.get(i)).append("\n");
            }
            return sb.toString();
        }
    }

    // ============ FIXED: Better existence check ============
    private boolean existsInRevision(int globalCloneId, int rev) {
        Set<Integer> revisions = cloneRevisionMap.get(globalCloneId);
        return revisions != null && revisions.contains(rev);
    }

    /**
     * Find the next available revision after the given revision.
     * Returns -1 if no next revision exists (handles non-consecutive revision
     * numbers)
     */
    private int findNextRevision(int currentRev) {
        for (int nextRev : availableRevisions) {
            if (nextRev > currentRev) {
                return nextRev;
            }
        }
        return -1; // No next revision found
    }

    /**
     * Find the next revision after currentRev where BOTH clones have data.
     * This is needed because clones are only recorded in revisions where they were
     * modified.
     */
    private int findNextRevisionWithBothClones(int g1, int g2, int currentRev) {
        Set<Integer> g1Revs = cloneRevisionMap.get(g1);
        Set<Integer> g2Revs = cloneRevisionMap.get(g2);

        if (g1Revs == null || g2Revs == null) {
            return -1;
        }

        // Find revisions where both clones have data, after currentRev
        for (int rev : availableRevisions) {
            if (rev > currentRev && g1Revs.contains(rev) && g2Revs.contains(rev)) {
                return rev;
            }
        }
        return -1;
    }

    // ============ Main SPCP Analysis (unchanged logic) ============
    public void runSPCP() throws IOException {
        String output = systemPath + File.separator + "SPCP_" + cloneType + "_" + granularity + ".csv";

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(output))) {
            // Write header
            bw.write("cloneType,granularity,gcid1,gcid2," +
                    "no_of_revision_paired,couplingStrength,latePropagationCount,dominantChangeCategory," +
                    "made_pair_in_revisions,gcid1_changed_revs,gcid2_changed_revs\n");

            System.out.println("=== SPCP Analysis Debug Info ===");
            System.out.println("Total clones loaded: " + cloneInfoMap.size());
            System.out.println("Total clone histories: " + historyMap.size());

            // Group clones by class ID
            Map<Integer, List<Integer>> classGroups = new HashMap<>();
            for (Clones c : cloneInfoMap.values()) {
                classGroups.computeIfAbsent(c.classId, k -> new ArrayList<>()).add(c.globalCloneId);
            }

            System.out.println("Total clone classes: " + classGroups.size());

            // Print class sizes for debugging
            for (Map.Entry<Integer, List<Integer>> entry : classGroups.entrySet()) {
                System.out.println("  Class " + entry.getKey() + ": " + entry.getValue().size() + " clones");
            }

            int totalPairs = 0;
            int pairsWithCoChanges = 0;
            int validSPCPPairs = 0;

            FragmentFetcher fetcher = new FileBasedFragmentFetcher();

            // Analyze each clone class
            for (Map.Entry<Integer, List<Integer>> entry : classGroups.entrySet()) {
                List<Integer> members = entry.getValue();
                Collections.sort(members);

                if (members.size() < 2) {
                    System.out.println("Class " + entry.getKey() + ": Only 1 clone, skipping");
                    continue;
                }

                System.out.println("\nProcessing class " + entry.getKey() + " with " + members.size() + " clones");

                // Analyze all pairs in this class
                for (int i = 0; i < members.size(); i++) {
                    for (int j = i + 1; j < members.size(); j++) {
                        totalPairs++;
                        int g1 = members.get(i);
                        int g2 = members.get(j);

                        CloneHistory h1 = historyMap.get(g1);
                        CloneHistory h2 = historyMap.get(g2);

                        if (h1 == null || h2 == null) {
                            System.out.println("  Pair (" + g1 + "," + g2 + "): Missing history");
                            continue;
                        }

                        // Find co-changes (revisions where both changed)
                        Set<Integer> coChange = new TreeSet<>(h1.changeRevs);
                        coChange.retainAll(h2.changeRevs);

                        if (coChange.isEmpty()) {
                            System.out.println("  Pair (" + g1 + "," + g2 + "): No co-changes " +
                                    "(g1 changed: " + h1.changeRevs + ", g2 changed: " + h2.changeRevs + ")");
                            continue;
                        }

                        pairsWithCoChanges++;
                        System.out.println("  Pair (" + g1 + "," + g2 + "): " + coChange.size() +
                                " co-changes: " + coChange);

                        // Analyze each co-change for SPCP
                        Set<Integer> validRevs = new TreeSet<>();
                        Map<ChangeCategory, Integer> categoryCounts = new HashMap<>();
                        for (ChangeCategory cat : ChangeCategory.values()) {
                            categoryCounts.put(cat, 0);
                        }

                        for (int r : coChange) {
                            // Find next revision where BOTH clones have data
                            int nextRev = findNextRevisionWithBothClones(g1, g2, r);

                            if (nextRev == -1) {
                                // No future revision with both clones - cannot verify similarity
                                System.out.println(
                                        "    Rev " + r + ": SKIPPED (no future rev with both clones to verify)");
                                categoryCounts.put(ChangeCategory.NONE,
                                        categoryCounts.get(ChangeCategory.NONE) + 1);
                                continue;
                            }

                            // Try to get fragments for similarity check
                            String before1 = fetcher.getFragmentTextAt(g1, r);
                            String before2 = fetcher.getFragmentTextAt(g2, r);
                            String after1 = fetcher.getFragmentTextAt(g1, nextRev);
                            String after2 = fetcher.getFragmentTextAt(g2, nextRev);

                            // If fragment fetch fails, we cannot verify similarity - skip
                            if (before1 == null || before2 == null || after1 == null || after2 == null) {
                                System.out.println(
                                        "    Rev " + r + " -> " + nextRev + ": SKIPPED (fragment fetch failed)");
                                categoryCounts.put(ChangeCategory.NONE,
                                        categoryCounts.get(ChangeCategory.NONE) + 1);
                                continue;
                            }

                            // Compute actual similarity
                            double simBefore = lcsSimilarity(before1, before2);
                            double simAfter = lcsSimilarity(after1, after2);

                            System.out.println("    Rev " + r + " -> " + nextRev + ": simBefore="
                                    + String.format("%.3f", simBefore) +
                                    ", simAfter=" + String.format("%.3f", simAfter));

                            // Categorize the change
                            ChangeCategory category = categorizeChange(simBefore, simAfter);
                            categoryCounts.put(category, categoryCounts.get(category) + 1);
                            System.out.println("    Rev " + r + ": Category=" + category);

                            // SPCP criterion: clones must REMAIN CLONES after change (simAfter >=
                            // threshold)
                            if (simAfter >= SIM_THRESHOLD) {
                                validRevs.add(r);
                                System.out.println("    Rev " + r + ": ✓ VALID SPCP (sim=" +
                                        String.format("%.3f", simAfter) + " >= " + SIM_THRESHOLD + ")");
                            } else {
                                System.out.println("    Rev " + r + ": ✗ NOT SPCP (sim=" +
                                        String.format("%.3f", simAfter) + " < " + SIM_THRESHOLD + ")");
                            }
                        }

                        if (validRevs.isEmpty()) {
                            System.out.println("  Pair (" + g1 + "," + g2 + "): No valid SPCP revisions");
                            continue;
                        }

                        // Calculate metrics
                        double couplingStrength = calculateCouplingStrength(h1, h2);
                        int latePropagationCount = detectLatePropagation(h1, h2);
                        ChangeCategory dominantCategory = getDominantCategory(categoryCounts);

                        // ONLY include pairs where dominant category is SIMILARITY_PRESERVING or
                        // RESYNCHRONIZING
                        // Exclude DIVERGING pairs - they are NOT Similarity-Preserving Change Patterns
                        if (dominantCategory == ChangeCategory.DIVERGING ||
                                dominantCategory == ChangeCategory.NONE) {
                            System.out.println("  Pair (" + g1 + "," + g2 +
                                    "): Excluded - dominant category is " + dominantCategory);
                            continue;
                        }

                        validSPCPPairs++;
                        System.out.println("  Pair (" + g1 + "," + g2 + "): ✓ " + validRevs.size() +
                                " VALID SPCP revisions, dominant=" + dominantCategory);

                        // Write results - only true SPCP clones
                        bw.write(String.join(",",
                                quote(cloneType),
                                quote(granularity),
                                String.valueOf(g1),
                                String.valueOf(g2),
                                String.valueOf(validRevs.size()),
                                String.format("%.4f", couplingStrength),
                                String.valueOf(latePropagationCount),
                                quote(dominantCategory.name()),
                                quote(validRevs.toString()),
                                quote(h1.changeRevs.toString()),
                                quote(h2.changeRevs.toString()))
                                + "\n");
                    }
                }
            }

            bw.flush();

            System.out.println("\n=== SPCP Summary ===");
            System.out.println("Total pairs checked: " + totalPairs);
            System.out.println("Pairs with co-changes: " + pairsWithCoChanges);
            System.out.println("Valid SPCP pairs found: " + validSPCPPairs);
            System.out.println("Output written to: " + output);
        }
    }

    // ============ Evolution Analysis Methods (unchanged) ============

    private ChangeCategory categorizeChange(double simBefore, double simAfter) {
        double delta = simAfter - simBefore;

        if (delta > RESYNC_THRESHOLD) {
            return ChangeCategory.RESYNCHRONIZING;
        } else if (delta >= -MAX_DROP && simAfter >= SIM_THRESHOLD) {
            return ChangeCategory.SIMILARITY_PRESERVING;
        } else {
            return ChangeCategory.DIVERGING;
        }
    }

    private double calculateCouplingStrength(CloneHistory h1, CloneHistory h2) {
        Set<Integer> intersection = new TreeSet<>(h1.changeRevs);
        intersection.retainAll(h2.changeRevs);

        Set<Integer> union = new TreeSet<>(h1.changeRevs);
        union.addAll(h2.changeRevs);

        if (union.isEmpty()) {
            return 0.0;
        }
        return (double) intersection.size() / union.size();
    }

    private int detectLatePropagation(CloneHistory h1, CloneHistory h2) {
        int latePropCount = 0;

        for (int rev1 : h1.changeRevs) {
            if (h2.changeRevs.contains(rev1))
                continue;

            for (int lag = 2; lag <= MAX_LAG_REVISIONS; lag++) {
                if (h2.changeRevs.contains(rev1 + lag)) {
                    latePropCount++;
                    break;
                }
            }
        }

        for (int rev2 : h2.changeRevs) {
            if (h1.changeRevs.contains(rev2))
                continue;

            for (int lag = 2; lag <= MAX_LAG_REVISIONS; lag++) {
                if (h1.changeRevs.contains(rev2 + lag)) {
                    latePropCount++;
                    break;
                }
            }
        }

        return latePropCount;
    }

    private ChangeCategory getDominantCategory(Map<ChangeCategory, Integer> categoryCounts) {
        ChangeCategory dominant = ChangeCategory.NONE;
        int maxCount = 0;

        for (Map.Entry<ChangeCategory, Integer> entry : categoryCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                dominant = entry.getKey();
            }
        }
        return dominant;
    }

    private String getEvolutionForecast(double couplingStrength, int latePropagationCount) {
        if (couplingStrength < 0.3) {
            return "INDEPENDENT";
        } else if (couplingStrength > 0.7 && latePropagationCount == 0) {
            return "CO_EVOLVING";
        } else if (latePropagationCount > 2) {
            return "UNSTABLE";
        }
        return "UNCERTAIN";
    }

    // ============ Helper Methods (unchanged) ============

    private static String quote(String s) {
        return "\"" + (s == null ? "" : s.replace("\"", "\"\"")) + "\"";
    }

    private static List<String> tokenize(String s) {
        if (s == null)
            return Collections.emptyList();
        s = s.replaceAll("[^A-Za-z0-9_]", " ");
        String[] toks = s.trim().split("\\s+");
        List<String> out = new ArrayList<>();
        for (String t : toks)
            if (!t.isEmpty())
                out.add(t);
        return out;
    }

    private static int lcsLength(List<String> a, List<String> b) {
        int n = a.size(), m = b.size();
        if (n == 0 || m == 0)
            return 0;
        int[] prev = new int[m + 1], cur = new int[m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++)
                cur[j] = a.get(i - 1).equals(b.get(j - 1)) ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
            int[] tmp = prev;
            prev = cur;
            cur = tmp;
            Arrays.fill(cur, 0);
        }
        return prev[m];
    }

    private static double lcsSimilarity(String a, String b) {
        List<String> ta = tokenize(a), tb = tokenize(b);
        if (ta.isEmpty() || tb.isEmpty())
            return 0.0;
        return (double) lcsLength(ta, tb) / Math.max(ta.size(), tb.size());
    }

    // ============ Data Classes ============

    private static class Clones {
        int globalCloneId;
        int classId;
        String filePath;
        int startLine, endLine;
        int addedInRev;
        Set<Integer> changeRevs;

        public Clones(int globalCloneId, int classId, String filePath,
                int startLine, int endLine, int addedInRev, Set<Integer> changeRevs) {
            this.globalCloneId = globalCloneId;
            this.classId = classId;
            this.filePath = filePath;
            this.startLine = startLine;
            this.endLine = endLine;
            this.addedInRev = addedInRev;
            this.changeRevs = changeRevs;
        }
    }

    private static class CloneHistory {
        Set<Integer> changeRevs;
        int addedInRev;

        public CloneHistory(Set<Integer> changeRevs, int addedInRev) {
            this.changeRevs = changeRevs;
            this.addedInRev = addedInRev;
        }
    }
}