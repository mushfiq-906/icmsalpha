/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JProgressBar;

/**
 * Links clones to their containing methods across revisions.
 * 
 * For each clone, finds the method that contains it by matching:
 * - File path (normalized)
 * - Clone line range [StartLine, EndLine] inside method range [start_line,
 * end_line]
 * 
 * @author Administrator
 */
public class CloneMethodLinker {

    private CommonParameters cp = new CommonParameters();
    private int cloneType;
    private String granularity;
    private String cloneTypeStr;
    private String granularityStr;
    private int startRev;
    private int endRev;

    /**
     * Constructor
     * 
     * @param cloneType   1, 2, or 3
     * @param granularity "f" for functions, "b" for blocks
     * @param startRev    starting revision number
     * @param endRev      ending revision number
     */
    public CloneMethodLinker(int cloneType, String granularity, int startRev, int endRev) {
        this.cloneType = cloneType;
        this.granularity = granularity;
        this.startRev = startRev;
        this.endRev = endRev;

        // Convert to folder names
        this.cloneTypeStr = "Type" + cloneType;
        this.granularityStr = granularity.equals("f") ? "Function" : "Block";
    }

    /**
     * Main method to link all clones to methods across revision range
     * 
     * @param progressBar optional progress bar to update
     */
    public void linkClonesToMethods(JProgressBar progressBar) {
        System.out.println("Starting clone-method linking...");
        System.out.println("Type: " + cloneTypeStr + ", Granularity: " + granularityStr);
        System.out.println("Revisions: " + startRev + " to " + endRev);

        for (int rev = startRev; rev <= endRev; rev++) {
            try {
                processRevision(rev);

                // Update progress bar if provided
                if (progressBar != null) {
                    int progress = (int) ((rev - startRev + 1) * 100.0 / (endRev - startRev + 1));
                    progressBar.setValue(progress);
                }
            } catch (IOException e) {
                System.err.println("Error processing revision " + rev + ": " + e.getMessage());
                e.printStackTrace();
            }
        }

        System.out.println("Clone-method linking complete!");
    }

    /**
     * Process a single revision - load clones and methods, match them, write output
     */
    private void processRevision(int revision) throws IOException {
        System.out.println("Processing revision " + revision + "...");

        // Load clones for this revision
        List<CloneInfo> clones = loadGlobalClones(revision);
        if (clones.isEmpty()) {
            System.out.println("  No clones found for revision " + revision);
            return;
        }

        // Load methods for this revision
        List<MethodInfo> methods = loadGlobalMethods(revision);
        if (methods.isEmpty()) {
            System.out.println("  No methods found for revision " + revision);
        }

        // Match clones to methods
        for (CloneInfo clone : clones) {
            matchCloneToMethod(clone, methods);
        }

        // Write output
        writeCloneInfo(revision, clones);

        System.out.println("  Processed " + clones.size() + " clones, " + methods.size() + " methods");
    }

    /**
     * Load global clones from file
     * Format: Revision | filePath | changeType | StartLine | EndLine | Nlines |
     * cloneId | classId | globalCloneId
     */
    private List<CloneInfo> loadGlobalClones(int revision) throws IOException {
        String path = getGlobalClonesPath(revision);
        List<CloneInfo> clones = new ArrayList<>();

        File file = new File(path);
        if (!file.exists()) {
            System.out.println("  Clone file not found: " + path);
            return clones;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine(); // Skip header
            String line;

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty())
                    continue;

                String[] parts = line.split("\\|");
                if (parts.length < 9)
                    continue;

                CloneInfo clone = new CloneInfo();
                clone.filePath = parts[1].trim();
                // parts[2] is changeType - not needed
                clone.startLine = Integer.parseInt(parts[3].trim());
                clone.endLine = Integer.parseInt(parts[4].trim());
                // parts[5] is nLines - not needed
                clone.cloneId = Integer.parseInt(parts[6].trim());
                clone.classId = Integer.parseInt(parts[7].trim());
                clone.globalCloneId = Integer.parseInt(parts[8].trim());

                clones.add(clone);
            }
        }

        return clones;
    }

    /**
     * Load global methods from file
     * Format: revision_number | method_name | method_signature | src | start_line |
     * end_line | methodId | globalMethodId | package_name | class_name
     */
    private List<MethodInfo> loadGlobalMethods(int revision) throws IOException {
        String path = cp.getGlobalMethodsPath(revision);
        List<MethodInfo> methods = new ArrayList<>();

        File file = new File(path);
        if (!file.exists()) {
            System.out.println("  Method file not found: " + path);
            return methods;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine(); // Skip header
            String line;

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty())
                    continue;

                String[] parts = line.split("\\|");
                if (parts.length < 8)
                    continue;

                MethodInfo method = new MethodInfo();
                method.revision = Integer.parseInt(parts[0].trim());
                method.methodName = parts[1].trim();
                method.methodSignature = parts[2].trim();
                // New column order: package_name and class_name come before src
                // GlobalMethods format: revision | name | signature | package | class | src |
                // start | end | methodId | globalMethodId
                if (parts.length > 9) {
                    method.packageName = parts[3].trim();
                    method.className = parts[4].trim();
                    method.filePath = parts[5].trim();
                    method.startLine = Integer.parseInt(parts[6].trim());
                    method.endLine = Integer.parseInt(parts[7].trim());
                    method.methodId = Integer.parseInt(parts[8].trim());
                    method.globalMethodId = Integer.parseInt(parts[9].trim());
                } else if (parts.length > 7) {
                    // Fallback for old format without package/class
                    method.filePath = parts[3].trim();
                    method.startLine = Integer.parseInt(parts[4].trim());
                    method.endLine = Integer.parseInt(parts[5].trim());
                    method.methodId = Integer.parseInt(parts[6].trim());
                    method.globalMethodId = Integer.parseInt(parts[7].trim());
                }

                methods.add(method);
            }
        }

        return methods;
    }

    /**
     * Match a clone to its containing method
     */
    private void matchCloneToMethod(CloneInfo clone, List<MethodInfo> methods) {
        for (MethodInfo method : methods) {
            // Check if file paths match (normalize slashes)
            if (!compareFilePaths(clone.filePath, method.filePath)) {
                continue;
            }

            // Check if clone is inside method
            if (clone.startLine >= method.startLine && clone.endLine <= method.endLine) {
                clone.globalMethodId = method.globalMethodId;
                clone.methodName = method.methodName;
                clone.methodSignature = method.methodSignature;
                clone.methodStart = method.startLine;
                clone.methodEnd = method.endLine;
                clone.packageName = method.packageName;
                clone.className = method.className;
                return; // Found the containing method
            }
        }

        // No method found - leave as NULL
        clone.globalMethodId = -1; // Will be written as NULL
        clone.methodName = null;
        clone.methodSignature = null;
        clone.methodStart = -1;
        clone.methodEnd = -1;
        clone.packageName = null;
        clone.className = null;
    }

    /**
     * Compare file paths, normalizing slashes and extracting relevant portion
     */
    private boolean compareFilePaths(String path1, String path2) {
        // Normalize slashes
        String normalized1 = path1.replace("\\", "/").toLowerCase();
        String normalized2 = path2.replace("\\", "/").toLowerCase();

        // Extract portion after "Revision_XXX/"
        String relevant1 = extractRelevantPath(normalized1);
        String relevant2 = extractRelevantPath(normalized2);

        return relevant1.equals(relevant2);
    }

    /**
     * Extract the file path portion after "Revision_XXX/"
     */
    private String extractRelevantPath(String fullPath) {
        int idx = fullPath.indexOf("revision_");
        if (idx != -1) {
            int slashIdx = fullPath.indexOf("/", idx);
            if (slashIdx != -1) {
                return fullPath.substring(slashIdx + 1);
            }
        }
        return fullPath;
    }

    /**
     * Write clone info to output file
     */
    private void writeCloneInfo(int revision, List<CloneInfo> clones) throws IOException {
        String path = getGlobalCloneInfoPath(revision);

        // Ensure directory exists
        File file = new File(path);
        file.getParentFile().mkdirs();

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            // Write header
            bw.write(
                    "globalCloneId | filePath | StartLine | EndLine | classId | globalMethodId | method_name | method_signature | methodStart | methodEnd | package_name | class_name\n");

            // Write each clone
            for (CloneInfo clone : clones) {
                String globalMethodIdStr = clone.globalMethodId == -1 ? "NULL" : String.valueOf(clone.globalMethodId);
                String methodNameStr = clone.methodName == null ? "NULL" : clone.methodName;
                String methodSignatureStr = clone.methodSignature == null ? "NULL" : clone.methodSignature;
                String methodStartStr = clone.methodStart == -1 ? "NULL" : String.valueOf(clone.methodStart);
                String methodEndStr = clone.methodEnd == -1 ? "NULL" : String.valueOf(clone.methodEnd);
                String packageNameStr = clone.packageName == null ? "" : clone.packageName;
                String classNameStr = clone.className == null ? "" : clone.className;

                String data = clone.globalCloneId + " | " + clone.filePath + " | " +
                        clone.startLine + " | " + clone.endLine + " | " + clone.classId + " | " +
                        globalMethodIdStr + " | " + methodNameStr + " | " + methodSignatureStr + " | " + methodStartStr
                        + " | " + methodEndStr + " | " + packageNameStr + " | " + classNameStr;

                bw.write(data + "\n");
            }
        }
    }

    /**
     * Get path to GlobalClones input file
     */
    private String getGlobalClonesPath(int revision) {
        return cp.WorkFolder + cp.getSubjectSystem() + "/Clones/" + cloneTypeStr + "/" +
                granularityStr + "/GlobalClones/" + revision + "_clones.txt";
    }

    /**
     * Get path to GlobalCloneInfo output file
     */
    private String getGlobalCloneInfoPath(int revision) {
        return cp.WorkFolder + cp.getSubjectSystem() + "/Clones/" + cloneTypeStr + "/" +
                granularityStr + "/GlobalCloneInfo/" + revision + "_clones.txt";
    }

    // Inner class to hold clone info
    private static class CloneInfo {
        String filePath;
        int startLine;
        int endLine;
        int cloneId;
        int classId;
        int globalCloneId;

        // Method info (populated after matching)
        int globalMethodId = -1;
        String methodName = null;
        String methodSignature = null;
        int methodStart = -1;
        int methodEnd = -1;
        String packageName = null;
        String className = null;
    }

    // Inner class to hold method info
    private static class MethodInfo {
        int revision;
        String methodName;
        String methodSignature;
        String filePath;
        int startLine;
        int endLine;
        int methodId;
        int globalMethodId;
        String packageName = "";
        String className = "";
    }

    // // Utility method for testing
    // public static void main(String[] args) {
    // // Example usage
    // int cloneType = 1;
    // String granularity = "f"; // functions
    // int startRev = 491;
    // int endRev = 500;
    //
    // CloneMethodLinker linker = new CloneMethodLinker(cloneType, granularity,
    // startRev, endRev);
    // linker.linkClonesToMethods(null);
    // }
}
