/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import javax.swing.JProgressBar;

/**
 *
 * @author Parzival
 */
class Pairs {
    int first;
    int second;

    Pairs(int first, int second) {
        this.first = first;
        this.second = second;
    }
}

public class CloneDetection {

    AccessPoint ap = new AccessPoint();
    int startRev = ap.firstGenRevision();
    LoadParam lp = new LoadParam();
    int endRev = lp.getLastRevNumber();
    List<CloneClass> cc = new ArrayList<>();
    CommonParameters cp = new CommonParameters();

    // Enhanced matching configuration (Dr. Mondal's methodology)
    // Thresholds now loaded from clone_config.properties via LoadParam
    private final double HIGH_SIMILARITY = lp.getHighSimilarity();
    private final int LINE_TOLERANCE = lp.getLineTolerance();

    /**
     * Get similarity threshold based on clone type (from external config)
     * Type 1: 0.99 (exact clones)
     * Type 2: 0.99 (exact after blind renaming)
     * Type 3: 0.70 (near-miss, matches NiCad threshold=0.3)
     */
    private double getSimilarityThreshold() {
        switch (choice) {
            case 1:
                return lp.getType1Threshold();
            case 2:
                return lp.getType2Threshold();
            case 3:
                return lp.getType3Threshold();
            default:
                return lp.getType3Threshold();
        }
    }

    private static int latestGlobalId = 0;
    List<Clones> prevClones = new ArrayList<>();
    List<Clones> currClones = new ArrayList<>();
    private int choice;
    private String granularity;
    private String gPath, cChoice;

    // Historical map: key = normalized "filePath_startLine" -> globalCloneId
    // This allows us to recognize clones that reappear after disappearing
    private static Map<String, Clones> historicalCloneMap = new HashMap<>();

    static List<Pairs> rgid = new ArrayList<>();

    CloneDetection(int choice, String granularity, int startRev, int endRev) {
        this.choice = choice;
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

    CloneDetection(int choice, String granularity) {
        this.choice = choice;
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

    private void saveMePlease() throws IOException {
        String rPath = cp.getRevisionToMaxGlobal(cChoice, granularity);

        BufferedWriter writer2 = new BufferedWriter(new FileWriter(rPath));
        for (Pairs p : rgid) {
            String ss = p.first + " | " + p.second;
            writer2.write(ss);
        }
        System.out.println(" Revision To Global Added Succesfully");
    }

    public void CloneMe(JProgressBar progressBar) {
        try {
            // storing the clones to database.
            // totalRevisionNumber = 700;
            // lowestRevision = 650;
            for (int i = startRev; i <= endRev; i++) {
                // storeClonesOfVersionToFile(i , 1);
                ap.extractAllClones(i, cChoice, granularity);

                /// store all the code clones
                int progress = (int) ((i - startRev) * 100.0 / endRev);
                // String progressText = "Clone Extract ..." + i + "%";
                // Update the JProgressBar on the Event Dispatch Thread
                // SwingUtilities.invokeLater(() -> {
                // progressBar.setValue(progress);
                // // progressBar.setString(progressText);
                //
                // });

            }

            // you can store them in to the database

        } catch (Exception e) {
            System.out.println("error. mainMethod. " + e);
        }
    }

    void waitAbit(int a) {
        try {
            Thread.sleep(a); // Simulate a delay
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    void globalCloneWithClassID() {
        try {

            // storing the clones to database.

            // lowestRevision = 370;//;totalRevisionNumber = 116;
            for (int i = startRev; i <= endRev; i++) {
                System.out.println("System Working Revision: " + i + "  Type:" + choice + "     Granularity:"
                        + granularity + "  ");
                if (i == startRev) {

                    // getFirstGlobalClassID(i);

                } else {
                    getRestGlobalClassID(i - 1, i);
                    break;
                }
                /// store all the code clones
            }

            // you can store them in to the database

        } catch (Exception e) {
            System.out.println("error. mainMethod. " + e);
        }

    }

    private void getFirstGlobalClassID(int revision) throws IOException {

        ap.setFirstGlobalClassId(revision, cChoice, granularity);
        System.out.println("Successfull");
        prevClones = ap.getFirstClones(revision, cChoice, granularity);
        latestGlobalId = prevClones.size();
        Pairs p = new Pairs(revision, latestGlobalId);
        rgid.add(p);

    }

    private void getRestGlobalClassID(int prevRevision, int currentRevision) {
        String basePath;

        basePath = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + cChoice + "/" + granularity
                + "/GlobalFinalClones/";

        String currentFileName = currentRevision + "_clones.txt";
        File outputFile = new File(basePath, currentFileName);
        // Ensure the directories exist
        outputFile.getParentFile().mkdirs();

        // Using try-with-resources to ensure the writer is closed properly
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile, true))) {
            // Check if the file is empty to write headers
            if (outputFile.length() == 0) {
                writer.write(
                        "Revision | filePath | changeType | StartLine | EndLine | cloneId | classId | globalCloneId \n");
            }

            // Assuming method calls to get clones return non-null and valid lists
            List<Clones> prevClones = ap.getFirstGlobalClassId(prevRevision, cChoice, granularity);

            List<Clones> currClones = ap.getGlobalClones(currentRevision, cChoice, granularity);

            // making hashmap in order to make gourp of clones respect to their class id
            Map<Integer, List<Clones>> prevClonesByClassId = new HashMap<>();
            Map<Integer, List<Clones>> currClonesByClassId = new HashMap<>();

            for (Clones clone : prevClones) {
                prevClonesByClassId.computeIfAbsent(clone.classId, k -> new ArrayList<>()).add(clone);
            }

            for (Map.Entry<Integer, List<Clones>> entry : prevClonesByClassId.entrySet()) {
                System.out.println("ClassId: " + entry.getKey());
                for (Clones clone : entry.getValue()) {
                    System.out.println(clone.filePath);
                }
                System.out.println();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        Pairs p = new Pairs(currentRevision, latestGlobalId);
        rgid.add(p);

    }

    class CloneClass {
        int fragment_count;
        List<CloneFragment> cloneFragment = new ArrayList<>();
    }

    class CloneFragment {
        String filePath = "";
        int startLine;
        int endLine;
    }

    public void globalClone(JProgressBar progressBar) throws IOException {
        try {
            // storing the clones to database.
            for (int i = startRev; i <= endRev; i++) {
                System.out.println("System Working Revision: " + i + "  Type:" + choice + "     Granularity:"
                        + granularity + "  ");
                if (i == startRev) {
                    getFirstGlobalClone(i);
                } else {
                    getRestGlobalClone(i - 1, i);
                    // print("Hello " + i);
                }

                int progress = (int) ((i - startRev) * 100.0 / endRev);

                /// store all the code clones
            }

            // you can store them in to the database

        } catch (Exception e) {
            System.out.println("error. mainMethod. " + e);
        }
        // saveMePlease();
        // evaluationClone();
    }

    /**
     * CLONE EVALUATION
     * 
     */
    private void evaluationClone() throws IOException {
        Integer[] cloneEval = new Integer[100000]; // Array to store change counts for each global clone ID
        int[] firstIntroduced = new int[100000]; // Array to store the first revision where each GCID was introduced

        for (int i = startRev; i <= endRev; i++) {
            String path = cp.getGlobalClonePath(cChoice, granularity, i);
            BufferedReader br = new BufferedReader(new FileReader(path));
            String line;
            br.readLine(); // Skip the first line
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");
                int gcid = Integer.parseInt(parts[7].trim()); // Assuming global clone ID is at index 7
                String changeType = parts[2].trim(); // Assuming changeType is at index 2
                int revision = Integer.parseInt(parts[0].trim());
                // Initialize cloneEval[gcid] if it's null
                if (cloneEval[gcid] == null) {
                    cloneEval[gcid] = 0;
                }

                // Increment count for 'M' and 'A' types
                if (changeType.equals("M")) {
                    cloneEval[gcid]++;

                    // Record the first revision where this GCID appears, if not already recorded
                    if (firstIntroduced[gcid] == 0) {
                        firstIntroduced[gcid] = revision;
                    }
                }
            }
            br.close(); // Close the BufferedReader
        }

        // Print and write results for global clone IDs that have been evaluated
        String spath = cp.getEvaluation(cChoice, granularity);
        spath = spath + "/cloneEvaluation.txt";
        BufferedWriter bw = new BufferedWriter(new FileWriter(spath));
        bw.write("GCID | ChangeCount | FirstIntroduced\n");
        for (int i = 1; i < cloneEval.length; i++) {
            if (cloneEval[i] != null && cloneEval[i] != 0) {
                String ss = i + " | " + cloneEval[i] + " | " + firstIntroduced[i] + "\n";
                bw.write(ss);
                System.out.println("GCID : " + i + "  ChangeCount : " + cloneEval[i] + "  FirstIntroduced : "
                        + firstIntroduced[i]);
            }
        }
        bw.close(); // Close the BufferedWriter
    }

    public void print(Object o) {
        System.out.println(o);
    }

    // for first revision
    public void getFirstGlobalClone(int revision) throws IOException {

        print("Debug setFirstGlobalClone");
        ap.setFirstGlobalClones(revision, cChoice, granularity);
        print("Debug getFirstGlobalClone");
        prevClones = ap.getFirstClones(revision, cChoice, granularity);

        // Clear and populate historical map from first revision
        historicalCloneMap.clear();
        for (Clones c : prevClones) {
            print(" ---> " + c.filePath + " " + c.revision);
            // Create key based on normalized path and line range
            String key = createHistoricalKey(c);
            historicalCloneMap.put(key, c);
        }

        latestGlobalId = prevClones.size();
        print("Historical map initialized with " + historicalCloneMap.size() + " clones");
        print(latestGlobalId);
        Pairs p = new Pairs(revision, latestGlobalId);
        rgid.add(p);

    }

    /**
     * Create a key for historical clone lookup based on file path and line range
     */
    private String createHistoricalKey(Clones c) {
        String normalizedPath = getPath(c.filePath);
        return normalizedPath + "_" + c.startLine + "_" + c.endLine;
    }

    private String extractFileName(String filePath) {
        return Paths.get(filePath).getFileName().toString();
    }

    public void getRestGlobalClone(int prevRevision, int currentRevision) throws IOException {
        String basePath = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + cChoice + "/" + granularity
                + "/GlobalClones/";
        String currentFileName = currentRevision + "_clones.txt";
        File outputFile = new File(basePath, currentFileName);

        // Ensure parent directories exist
        outputFile.getParentFile().mkdirs();

        // Load offsets safely for CURRENT revision
        List<OffsetRevision> offsets = new ArrayList<>();
        try {
            offsets = ap.getOffsetByRevision(currentRevision);
            System.out.println("Offset file loaded for revision " + currentRevision);
        } catch (FileNotFoundException e) {
            System.out.println("No offset file for revision " + currentRevision + ". Proceeding without offsets.");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Organize offsets by current file path (prevPath used as key) for fast access
        Map<String, List<OffsetRevision>> fileOffsetMap = offsets.stream()
                .collect(Collectors.groupingBy(o -> o.prevPath));

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile, true))) {
            // Write header if file is empty
            if (outputFile.length() == 0) {
                writer.write(
                        "Revision | filePath | changeType | StartLine | EndLine | Nlines | cloneId | classId | globalCloneId | Similarity\n");
            }

            // Load previous and current clones
            List<Clones> prevClones = ap.getFirstClones(prevRevision, cChoice, granularity);
            List<Clones> currClones = ap.getClones(currentRevision, cChoice, granularity);

            for (Clones currClone : currClones) {
                String cFilePath = getPath(currClone.filePath);

                // 1. Get relevant offsets for this file (for similarity and changeType check)
                List<OffsetRevision> relevantOffsets = fileOffsetMap.getOrDefault(cFilePath, Collections.emptyList());

                // Check if currClone overlaps any change and assign changeType
                for (OffsetRevision o : relevantOffsets) {
                    // Overlap check: any intersection between clone lines and offset lines
                    if (!(currClone.endLine < o.currStartLine || currClone.startLine > o.currEndLine)) {
                        currClone.cloneType = o.changeType; // Assign changeType
                        break; // If matched, no need to check further
                    }
                }

                boolean found = false;
                Iterator<Clones> iterator = prevClones.iterator();
                while (iterator.hasNext()) {
                    Clones prevClone = iterator.next();
                    String pFilePath = getPath(prevClone.filePath);

                    // Check if files match: exact path OR same filename (handles renames)
                    String pFileName = extractFileName(pFilePath);
                    String cFileName = extractFileName(cFilePath);
                    boolean pathMatches = pFilePath.equals(cFilePath);
                    boolean fileNameMatches = !pathMatches && pFileName.equals(cFileName);

                    if (pathMatches || fileNameMatches) {
                        // Calculate similarity
                        double similarity = getCodeSimilarity(prevClone, currClone);

                        // When only filename matches (file possibly moved/renamed),
                        // require higher similarity to avoid false matches with
                        // same-named files in different folders
                        double requiredThreshold = pathMatches ? getSimilarityThreshold() : HIGH_SIMILARITY;

                        if (similarity >= requiredThreshold) {
                            // Check offset similarity for genealogy tracking
                            for (OffsetRevision o : relevantOffsets) {
                                if (currClone.startLine >= o.prevStartLine && currClone.endLine <= o.prevEndLine) {
                                    // Store genealogy info if needed
                                }
                            }

                            // Assign globalCloneId from previous clone
                            currClone.globalCloneId = prevClone.globalCloneId;
                            iterator.remove();
                            found = true;

                            // Log match info
                            if (fileNameMatches) {
                                System.out.println("MATCHED (file moved, similarity="
                                        + String.format("%.2f", similarity)
                                        + "): gcid=" + currClone.globalCloneId + " " + pFilePath + " -> " + cFilePath);
                            } else if (similarity < HIGH_SIMILARITY) {
                                System.out.println("MATCHED (similarity=" + String.format("%.2f", similarity)
                                        + "): gcid=" + currClone.globalCloneId);
                            }
                            break;
                        }
                    }
                }

                // If not found in previous revision, check historical map for reappearing
                // clones
                if (!found) {
                    String histKey = createHistoricalKey(currClone);
                    Clones historicalClone = historicalCloneMap.get(histKey);

                    if (historicalClone != null) {
                        // Found exact match in history by key (file path + line range)
                        // This means the clone reappeared at the same location
                        currClone.globalCloneId = historicalClone.globalCloneId;
                        found = true;
                        System.out.println("REAPPEARED FROM HISTORY (exact match): gcid=" + currClone.globalCloneId
                                + " file=" + extractFileName(currClone.filePath));
                    } else {
                        // Try fuzzy matching - same file, similar line range (within tolerance)
                        // cFilePath is already defined earlier in the method
                        for (Map.Entry<String, Clones> entry : historicalCloneMap.entrySet()) {
                            Clones histClone = entry.getValue();
                            String hFilePath = getPath(histClone.filePath);

                            // Same file path
                            if (cFilePath.equals(hFilePath)) {
                                // Check if line ranges are within tolerance
                                int startDiff = Math.abs(currClone.startLine - histClone.startLine);
                                int endDiff = Math.abs(currClone.endLine - histClone.endLine);

                                if (startDiff <= LINE_TOLERANCE && endDiff <= LINE_TOLERANCE) {
                                    currClone.globalCloneId = histClone.globalCloneId;
                                    found = true;
                                    System.out.println(
                                            "REAPPEARED FROM HISTORY (fuzzy match): gcid=" + currClone.globalCloneId
                                                    + " file=" + extractFileName(currClone.filePath)
                                                    + " startDiff=" + startDiff + " endDiff=" + endDiff);
                                    break;
                                }
                            }
                        }
                    }
                }

                if (!found) {
                    latestGlobalId++;
                    currClone.globalCloneId = latestGlobalId;
                    System.out.println("NEW CLONE INTRODUCED: gcid=" + latestGlobalId
                            + " file=" + extractFileName(currClone.filePath));
                }

                // Always add/update the clone in historical map for future lookups
                String histKey = createHistoricalKey(currClone);
                historicalCloneMap.put(histKey, currClone);

                // 2. Save clone info to file (now including similarity)
                String data = currentRevision + " | " + currClone.filePath + " | " + currClone.cloneType + " | "
                        + currClone.startLine + " | " + currClone.endLine + " | " + currClone.nLines
                        + " | " + currClone.cloneId + " | " + currClone.classId + " | " + currClone.globalCloneId
                        + " | " + currClone.similarity;
                writer.write(data + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Update latest global ID mapping
        rgid.add(new Pairs(currentRevision, latestGlobalId));
    }

    /**
     * Enhanced clone matching using similarity threshold instead of exact match
     * Returns true if similarity >= SIMILARITY_THRESHOLD (default 70%)
     */
    private boolean isItSameCode(Clones pClone, Clones cClone) {
        String cCodeFile = cp.getCloneDir() + "/" + cClone.filePath;
        String pCodeFile = cp.getCloneDir() + "/" + pClone.filePath;

        String cCode = loadCodeFragment(cCodeFile, cClone.startLine, cClone.endLine);
        String pCode = loadCodeFragment(pCodeFile, pClone.startLine, pClone.endLine);

        if (cCode == null || pCode == null) {
            return false;
        }

        // Use similarity-based matching with type-specific threshold
        double similarity = calculateSimilarity(pCode, cCode);
        return similarity >= getSimilarityThreshold();
    }

    /**
     * Enhanced version that returns similarity score for detailed tracking
     */
    private double getCodeSimilarity(Clones pClone, Clones cClone) {
        String cCodeFile = cp.getCloneDir() + "/" + cClone.filePath;
        String pCodeFile = cp.getCloneDir() + "/" + pClone.filePath;

        String cCode = loadCodeFragment(cCodeFile, cClone.startLine, cClone.endLine);
        String pCode = loadCodeFragment(pCodeFile, pClone.startLine, pClone.endLine);

        if (cCode == null || pCode == null) {
            return 0.0;
        }

        return calculateSimilarity(pCode, cCode);
    }

    /**
     * Calculate similarity between two code fragments using LCS (Longest Common
     * Subsequence)
     * Based on token-level comparison for better accuracy
     */
    private double calculateSimilarity(String code1, String code2) {
        List<String> tokens1 = tokenize(code1);
        List<String> tokens2 = tokenize(code2);

        if (tokens1.isEmpty() || tokens2.isEmpty()) {
            return 0.0;
        }

        int lcs = lcsLength(tokens1, tokens2);
        return (double) lcs / Math.max(tokens1.size(), tokens2.size());
    }

    /**
     * Tokenize code into meaningful tokens (identifiers, keywords, operators)
     */
    private List<String> tokenize(String code) {
        if (code == null)
            return Collections.emptyList();
        // Remove non-alphanumeric except underscores, split by whitespace
        String normalized = code.replaceAll("[^A-Za-z0-9_]", " ");
        String[] tokens = normalized.trim().split("\\s+");
        List<String> result = new ArrayList<>();
        for (String t : tokens) {
            if (!t.isEmpty()) {
                result.add(t);
            }
        }
        return result;
    }

    /**
     * Calculate LCS length using dynamic programming (space-optimized)
     */
    private int lcsLength(List<String> a, List<String> b) {
        int n = a.size(), m = b.size();
        if (n == 0 || m == 0)
            return 0;

        int[] prev = new int[m + 1];
        int[] cur = new int[m + 1];

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (a.get(i - 1).equals(b.get(j - 1))) {
                    cur[j] = prev[j - 1] + 1;
                } else {
                    cur[j] = Math.max(prev[j], cur[j - 1]);
                }
            }
            int[] tmp = prev;
            prev = cur;
            cur = tmp;
            Arrays.fill(cur, 0);
        }
        return prev[m];
    }

    /**
     * Check if line positions are within tolerance (handles code movement)
     */
    private boolean isLinePositionSimilar(Clones pClone, Clones cClone) {
        int startDiff = Math.abs(pClone.startLine - cClone.startLine);
        int endDiff = Math.abs(pClone.endLine - cClone.endLine);
        return startDiff <= LINE_TOLERANCE && endDiff <= LINE_TOLERANCE;
    }

    // ================ Path and File Utilities ================

    /**
     * Normalize file path for comparison (removes all Revision_* prefixes)
     * Example: "Revision_300/Revision_300/asp.c" -> "asp.c"
     */
    private static String getPath(String filePath) {
        // Handle both / and \ separators
        String normalized = filePath.replace("\\", "/");

        // Remove all Revision_* prefixes from the path
        // Pattern matches: Revision_123/ at any point in the path
        String result = normalized.replaceAll("Revision_\\d+/", "");

        // If result is empty (shouldn't happen), return original
        if (result.isEmpty()) {
            return normalized;
        }

        return result;
    }

    private String loadCodeFragment(String filePath, int startLine, int endLine) {
        StringBuilder codeFragment = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            int currentLine = 1;
            while ((line = reader.readLine()) != null) {
                if (currentLine >= startLine && currentLine <= endLine) {
                    codeFragment.append(line).append("\n");
                }
                currentLine++;
            }
        } catch (FileNotFoundException e) {
            // File not found, handle it by returning null
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null; // Return null in case of other IO errors
        }
        return codeFragment.toString();
    }

    // Here we will check for the clone if it is a
    public boolean isMicroCloneClass(CloneClass clone_class) {
        for (int i = 0; i < clone_class.cloneFragment.size(); i++) {
            int endline = clone_class.cloneFragment.get(i).endLine;
            int startline = clone_class.cloneFragment.get(i).startLine;

            if (endline - startline + 1 <= 4) {
                return true;
            }
        }
        return false;
    }

    public int classExistence(CloneClass clone_class) {
        for (int i = 0; i < cc.size(); i++) {
            if (cc.get(i).cloneFragment.size() == clone_class.cloneFragment.size()) {
                boolean allFragmentsMatch = true;

                for (int j = 0; j < clone_class.cloneFragment.size(); j++) {
                    boolean fragmentMatchFound = false;

                    for (int l = 0; l < cc.get(i).cloneFragment.size(); l++) {
                        if (clone_class.cloneFragment.get(j).filePath.equals(cc.get(i).cloneFragment.get(l).filePath) &&
                                clone_class.cloneFragment.get(j).startLine == cc.get(i).cloneFragment.get(l).startLine
                                &&
                                clone_class.cloneFragment.get(j).endLine == cc.get(i).cloneFragment.get(l).endLine) {
                            fragmentMatchFound = true;
                            break;
                        }
                    }

                    if (!fragmentMatchFound) {
                        allFragmentsMatch = false;
                        break;
                    }
                }

                if (allFragmentsMatch) {
                    return 1;
                }
            }
        }
        return 0;
    }

    public void mapClonesToMethods(int cloneType) throws IOException {
        int highestRevision = lp.getLastRevNumber();

        for (int i = startRev; i <= highestRevision; i++) {
            System.out.println("Mapping clones to methods for revision " + i);
            String basePath;

            basePath = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/FinalClones/";

            String currentFileName = i + "_clones.txt";
            File outputFile = new File(basePath, currentFileName);

            // Ensure the directories exist
            outputFile.getParentFile().mkdirs();

            // Using try-with-resources to ensure the writer is closed properly
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile, true))) {
                // Check if the file is empty to write headers
                if (outputFile.length() == 0) {
                    writer.write("revision | cloneId | filePath | startLine | endLine | globalCloneId\n");
                }
                List<SingleMethod> methods = ap.getFinalMethods(i);
                List<Clones> clones = ap.getFinalClones(i);

                for (Clones clone : clones) {
                    // clone.cloneType = cloneType;
                    String cloneFilePath = ap.relevantPath(clone.filePath);
                    int cloneStartLine = clone.startLine;
                    int cloneEndLine = clone.endLine;

                    for (SingleMethod method : methods) {
                        String methodFilePath = ap.relevantPath(method.filePath);
                        int methodStartLine = method.startLine;
                        int methodEndLine = method.endLine;
                        // print(cloneFilePath.equalsIgnoreCase(methodFilePath) + " "+cloneFilePath+ "
                        // "+methodFilePath );
                        // Check if the clone fragment is fully within the method
                        if (compareFilePaths(methodFilePath, cloneFilePath) &&
                                cloneStartLine >= methodStartLine &&
                                cloneEndLine <= methodEndLine) {

                            clone.methodId = method.methodId;
                            break; // Once mapped to a method, break the inner loop
                        }
                    }

                    // Save results to file
                    String data = i + " | " + clone.cloneId + " | " + clone.filePath + " | " + clone.startLine + " | "
                            + clone.endLine + " | " + clone.globalCloneId + " | " + clone.methodId;
                    writer.write(data + "\n");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static boolean compareFilePaths(String path1, String path2) {
        // Normalize paths by converting to lowercase and replacing backslashes with
        // forward slashes
        String normalizedPath1 = path1.toLowerCase().replace("\\", "/");
        String normalizedPath2 = path2.toLowerCase().replace("\\", "/");

        // Compare the normalized paths
        return normalizedPath1.equals(normalizedPath2);
    }

}
