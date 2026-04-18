/*
 * MLDatasetExporter.java
 * 
 * Exports training data for ML model to forecast independent evolution of code clones.
 * Outputs: Type1.csv, Type2.csv, Type3.csv
 * 
 * Features exported:
 * - Clone-level: global_clone_id, clone_type, clone_size, lifespan, age, change_count, stability_index, author
 * - Class-level: class_id, class_size, same_file_ratio, avg_similarity
 * - Method-level: global_method_id, method_name, method_size
 * - Evolution: coupling_strength, independent_change_count, co_change_count, independent_evolution_score, spcp_pattern
 * - Target: will_evolve_independently (label)
 */
package com.mycompany.icmsalpha;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MLDatasetExporter {

    private CommonParameters cp = new CommonParameters();
    private LoadParam lp = new LoadParam();
    private int startRev, endRev;
    private String granularity;

    // Data holders
    private Map<Integer, CloneRecord> cloneRecords = new HashMap<>(); // globaCloneId -> record
    private Map<Integer, List<Integer>> classToClones = new HashMap<>(); // classId -> list of gcids
    private Map<Integer, MethodInfo> cloneToMethod = new HashMap<>(); // globalCloneId -> MethodInfo
    private Map<Integer, SPCPInfo> clonePairSPCP = new HashMap<>(); // gcid -> SPCP info (averaged from pairs)
    private Map<Integer, String> revisionAuthor = new HashMap<>(); // revision -> author

    // Inner class to hold clone data
    private static class CloneRecord {
        int globalCloneId;
        int classId;
        int cloneType; // 1, 2, or 3
        String filePath;
        int startLine, endLine;
        int cloneSize;

        // Genealogy data
        int addedInRev;
        int deletedInRev;
        int lifespan;
        int changeCount;
        double stabilityIndex;
        double independentEvolutionScore;
        int independentChangeCount;
        int coChangeCount;
        List<String> authors = new ArrayList<>();

        // Computed
        int age; // current_rev - addedInRev
    }

    private static class MethodInfo {
        int globalMethodId;
        String methodName;
        int methodStart, methodEnd;
        int methodSize;
    }

    private static class SPCPInfo {
        double couplingStrength;
        int latePropagationCount;
        String evolutionForecast; // INDEPENDENT, CO_EVOLVING, UNSTABLE, UNCERTAIN
        String dominantCategory; // SIMILARITY_PRESERVING, DIVERGING, RESYNCHRONIZING
    }

    public MLDatasetExporter(int startRev, int endRev, String granularity) {
        this.startRev = startRev;
        this.endRev = endRev;
        this.granularity = granularity.equals("f") ? "Function" : "Block";
    }

    /**
     * Export ML datasets for all three clone types
     */
    public void exportAll() throws IOException {
        System.out.println("=== ML Dataset Export Started ===");

        // Export for each clone type
        for (int type = 1; type <= 3; type++) {
            exportForCloneType(type);
        }

        System.out.println("=== ML Dataset Export Complete ===");
    }

    /**
     * Export dataset for a specific clone type
     */
    public void exportForCloneType(int cloneType) throws IOException {
        String typeStr = "Type" + cloneType;
        System.out.println("\nExporting " + typeStr + "...");

        // Clear data from previous type
        cloneRecords.clear();
        classToClones.clear();
        cloneToMethod.clear();
        clonePairSPCP.clear();
        revisionAuthor.clear();

        // Step 1: Load genealogy data (has most features already computed)
        loadGenealogyData(typeStr);

        // Step 2: Load method linking data
        loadMethodData(typeStr);

        // Step 3: Load SPCP data
        loadSPCPData(typeStr);

        // Step 4: Compute class features
        computeClassFeatures();

        // Step 5: Export to CSV
        String outputPath = cp.WorkFolder + cp.getSubjectSystem() + "/Datasets/" + typeStr + ".csv";
        exportToCSV(outputPath, cloneType);

        System.out.println("Exported " + cloneRecords.size() + " clones to " + outputPath);
    }

    /**
     * Load genealogy data from CloneGenealogyAnalysis output
     */
    private void loadGenealogyData(String typeStr) throws IOException {
        String path = cp.getGenealogyPath(typeStr, granularity);

        if (!Files.exists(Paths.get(path))) {
            System.out.println("WARNING: Genealogy file not found: " + path);
            return;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String header = br.readLine(); // Skip header
            String line;

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty())
                    continue;

                // Parse CSV:
                // cloneType,granularity,globalCloneId,filepath,authors,addedInRev,deletedInRev,lifespan,
                // totalChanges,totalUnchanged,stabilityIndex,changeProneness,independentEvolutionScore,
                // coChangeCount,independentChangeCount,changeRevs,unchangedRevs
                String[] parts = parseCSVLine(line);
                if (parts.length < 15)
                    continue;

                CloneRecord rec = new CloneRecord();
                rec.cloneType = Integer.parseInt(typeStr.replace("Type", ""));
                rec.globalCloneId = Integer.parseInt(parts[2].trim());
                rec.filePath = parts[3].trim();

                // Parse authors (may be quoted)
                String authorStr = parts[4].replace("\"", "").trim();
                rec.authors = Arrays.asList(authorStr.split(",\\s*"));

                rec.addedInRev = Integer.parseInt(parts[5].trim());
                String deletedStr = parts[6].trim();
                rec.deletedInRev = deletedStr.equals("Still Active") ? 0 : Integer.parseInt(deletedStr);
                rec.lifespan = Integer.parseInt(parts[7].trim());
                rec.changeCount = Integer.parseInt(parts[8].trim());
                rec.stabilityIndex = Double.parseDouble(parts[10].trim());
                rec.independentEvolutionScore = Double.parseDouble(parts[12].trim());
                rec.coChangeCount = Integer.parseInt(parts[13].trim());
                rec.independentChangeCount = Integer.parseInt(parts[14].trim());

                // Age = endRev - addedInRev
                rec.age = endRev - rec.addedInRev;

                cloneRecords.put(rec.globalCloneId, rec);
            }
        }

        System.out.println("  Loaded " + cloneRecords.size() + " clones from genealogy");
    }

    /**
     * Load method linking data from GlobalCloneInfo
     */
    private void loadMethodData(String typeStr) throws IOException {
        // Try each revision
        for (int rev = startRev; rev <= endRev; rev++) {
            String path = cp.WorkFolder + cp.getSubjectSystem() + "/Clones/" + typeStr + "/" +
                    granularity + "/GlobalCloneInfo/" + rev + "_clones.txt";

            if (!Files.exists(Paths.get(path)))
                continue;

            try (BufferedReader br = new BufferedReader(new FileReader(path))) {
                br.readLine(); // Skip header
                String line;

                while ((line = br.readLine()) != null) {
                    if (line.trim().isEmpty())
                        continue;

                    // Format: globalCloneId | filePath | StartLine | EndLine | classId |
                    // globalMethodId | method_name | methodStart | methodEnd
                    String[] parts = line.split("\\|");
                    if (parts.length < 9)
                        continue;

                    int gcid = Integer.parseInt(parts[0].trim());
                    int classId = Integer.parseInt(parts[4].trim());
                    String globalMethodIdStr = parts[5].trim();
                    String methodName = parts[6].trim();
                    String methodStartStr = parts[7].trim();
                    String methodEndStr = parts[8].trim();

                    // Update clone record with classId and size
                    CloneRecord rec = cloneRecords.get(gcid);
                    if (rec != null) {
                        rec.classId = classId;
                        rec.startLine = Integer.parseInt(parts[2].trim());
                        rec.endLine = Integer.parseInt(parts[3].trim());
                        rec.cloneSize = rec.endLine - rec.startLine + 1;

                        // Track class membership
                        classToClones.computeIfAbsent(classId, k -> new ArrayList<>()).add(gcid);
                    }

                    // Store method info if exists
                    if (!globalMethodIdStr.equals("NULL") && !methodName.equals("NULL")) {
                        MethodInfo info = new MethodInfo();
                        info.globalMethodId = Integer.parseInt(globalMethodIdStr);
                        info.methodName = methodName;
                        info.methodStart = methodStartStr.equals("NULL") ? 0 : Integer.parseInt(methodStartStr);
                        info.methodEnd = methodEndStr.equals("NULL") ? 0 : Integer.parseInt(methodEndStr);
                        info.methodSize = info.methodEnd - info.methodStart + 1;
                        cloneToMethod.put(gcid, info);
                    }
                }
            }
        }

        System.out.println("  Loaded method info for " + cloneToMethod.size() + " clones");
    }

    /**
     * Load SPCP analysis data
     */
    private void loadSPCPData(String typeStr) throws IOException {
        String path = cp.WorkFolder + cp.getSubjectSystem() + "/SPCP_" + typeStr + "_" + granularity + ".csv";

        if (!Files.exists(Paths.get(path))) {
            System.out.println("  WARNING: SPCP file not found: " + path);
            return;
        }

        // Map to aggregate SPCP data per clone (from all its pairs)
        Map<Integer, List<SPCPInfo>> spcpByClone = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine(); // Skip header
            String line;

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty())
                    continue;

                // Format:
                // cloneType,granularity,gcid1,gcid2,no_of_revision_paired,couplingStrength,
                // latePropagationCount,dominantChangeCategory,evolutionForecast,...
                String[] parts = parseCSVLine(line);
                if (parts.length < 9)
                    continue;

                int gcid1 = Integer.parseInt(parts[2].trim());
                int gcid2 = Integer.parseInt(parts[3].trim());

                SPCPInfo info = new SPCPInfo();
                info.couplingStrength = Double.parseDouble(parts[5].trim());
                info.latePropagationCount = Integer.parseInt(parts[6].trim());
                info.dominantCategory = parts[7].replace("\"", "").trim();
                info.evolutionForecast = parts[8].replace("\"", "").trim();

                // Store for both clones in pair
                spcpByClone.computeIfAbsent(gcid1, k -> new ArrayList<>()).add(info);
                spcpByClone.computeIfAbsent(gcid2, k -> new ArrayList<>()).add(info);
            }
        }

        // Average SPCP data per clone
        for (Map.Entry<Integer, List<SPCPInfo>> entry : spcpByClone.entrySet()) {
            int gcid = entry.getKey();
            List<SPCPInfo> infos = entry.getValue();

            SPCPInfo avg = new SPCPInfo();
            double sumCoupling = 0;
            int sumLateProp = 0;
            Map<String, Integer> categoryCounts = new HashMap<>();
            Map<String, Integer> forecastCounts = new HashMap<>();

            for (SPCPInfo i : infos) {
                sumCoupling += i.couplingStrength;
                sumLateProp += i.latePropagationCount;
                categoryCounts.merge(i.dominantCategory, 1, Integer::sum);
                forecastCounts.merge(i.evolutionForecast, 1, Integer::sum);
            }

            avg.couplingStrength = sumCoupling / infos.size();
            avg.latePropagationCount = sumLateProp / infos.size();
            avg.dominantCategory = getMostCommon(categoryCounts);
            avg.evolutionForecast = getMostCommon(forecastCounts);

            clonePairSPCP.put(gcid, avg);
        }

        System.out.println("  Loaded SPCP data for " + clonePairSPCP.size() + " clones");
    }

    private String getMostCommon(Map<String, Integer> counts) {
        return counts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("UNKNOWN");
    }

    /**
     * Compute class-level features
     */
    private void computeClassFeatures() {
        // Already built classToClones map during method loading
        System.out.println("  Computed class features for " + classToClones.size() + " classes");
    }

    /**
     * Compute same_file_ratio for a class
     */
    private double computeSameFileRatio(int classId) {
        List<Integer> members = classToClones.get(classId);
        if (members == null || members.size() <= 1)
            return 1.0;

        Map<String, Integer> fileCounts = new HashMap<>();
        for (int gcid : members) {
            CloneRecord rec = cloneRecords.get(gcid);
            if (rec != null && rec.filePath != null) {
                fileCounts.merge(rec.filePath, 1, Integer::sum);
            }
        }

        if (fileCounts.isEmpty())
            return 1.0;

        // Ratio = largest group / total
        int maxInSameFile = fileCounts.values().stream().max(Integer::compare).orElse(1);
        return (double) maxInSameFile / members.size();
    }

    /**
     * Export to CSV file
     */
    private void exportToCSV(String outputPath, int cloneType) throws IOException {
        // Ensure directory exists
        new File(outputPath).getParentFile().mkdirs();

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath))) {
            // Header
            bw.write("global_clone_id,clone_type,clone_size,lifespan,age,change_count,stability_index," +
                    "author,class_id,class_size,same_file_ratio," +
                    "global_method_id,method_name,method_size," +
                    "coupling_strength,independent_change_count,co_change_count," +
                    "independent_evolution_score,late_propagation_count,spcp_pattern," +
                    "dataset_split,will_evolve_independently\n");

            // Calculate train/test split point (70% training)
            int trainEndRev = startRev + (int) ((endRev - startRev) * 0.7);

            for (CloneRecord rec : cloneRecords.values()) {
                // Get class features
                int classSize = classToClones.containsKey(rec.classId) ? classToClones.get(rec.classId).size() : 1;
                double sameFileRatio = computeSameFileRatio(rec.classId);

                // Get method features
                MethodInfo method = cloneToMethod.get(rec.globalCloneId);
                String globalMethodId = method != null ? String.valueOf(method.globalMethodId) : "NULL";
                String methodName = method != null ? method.methodName : "NULL";
                String methodSize = method != null ? String.valueOf(method.methodSize) : "0";

                // Get SPCP features
                SPCPInfo spcp = clonePairSPCP.get(rec.globalCloneId);
                double couplingStrength = spcp != null ? spcp.couplingStrength : 0.0;
                int latePropCount = spcp != null ? spcp.latePropagationCount : 0;
                String spcpPattern = spcp != null ? spcp.dominantCategory : "UNKNOWN";

                // Primary author (first in list)
                String author = rec.authors.isEmpty() ? "UNKNOWN" : rec.authors.get(0);

                // Dataset split
                String split = rec.addedInRev <= trainEndRev ? "TRAIN" : "TEST";

                // Target variable: will_evolve_independently
                // Based on independent_evolution_score > 0.5 (more independent than
                // co-evolving)
                int target = rec.independentEvolutionScore > 0.5 ? 1 : 0;

                // Write row
                bw.write(String.format("%d,%d,%d,%d,%d,%d,%.4f,%s,%d,%d,%.4f,%s,%s,%s,%.4f,%d,%d,%.4f,%d,%s,%s,%d\n",
                        rec.globalCloneId,
                        rec.cloneType,
                        rec.cloneSize,
                        rec.lifespan,
                        rec.age,
                        rec.changeCount,
                        rec.stabilityIndex,
                        quote(author),
                        rec.classId,
                        classSize,
                        sameFileRatio,
                        globalMethodId,
                        quote(methodName),
                        methodSize,
                        couplingStrength,
                        rec.independentChangeCount,
                        rec.coChangeCount,
                        rec.independentEvolutionScore,
                        latePropCount,
                        quote(spcpPattern),
                        split,
                        target));
            }
        }
    }

    private String quote(String s) {
        if (s == null)
            return "NULL";
        // Escape quotes and wrap in quotes if contains comma
        s = s.replace("\"", "\"\"");
        if (s.contains(",") || s.contains("\"")) {
            return "\"" + s + "\"";
        }
        return s;
    }

    /**
     * Parse CSV line handling quoted fields
     */
    private String[] parseCSVLine(String line) {
        List<String> parts = new ArrayList<>();
        boolean inQuotes = false;
        StringBuilder current = new StringBuilder();

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);

            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                parts.add(current.toString());
                current = new StringBuilder();
            } else {
                current.append(c);
            }
        }
        parts.add(current.toString());

        return parts.toArray(new String[0]);
    }

    /**
     * Main method for testing
     */
    public static void main(String[] args) {
        try {
            LoadParam lp = new LoadParam();
            int startRev = lp.getFirstRevNumber();
            int endRev = lp.getLastRevNumber();

            MLDatasetExporter exporter = new MLDatasetExporter(startRev, endRev, "f");
            exporter.exportAll();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
