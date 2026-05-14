/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

/**
 *
 * @author Administrator
 */

import static com.mycompany.icmsalpha.FileUtil.getFileContentByLineRange;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import java.util.List;
import java.nio.file.Files;
import java.nio.file.Paths;
import com.opencsv.exceptions.CsvValidationException;
import difflib.DiffUtils;
import difflib.Patch;
import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;

public class MethodGenealogyAnalysis {
    int maxGlobalCloneID = -1;
    int matchThreshold = 70;
    LoadParam lp = new LoadParam();
    String prefix = lp.getCloneDir();
    int totalRevision = lp.getLastRevNumber();
    List<SingleMethod> previousMethods = new ArrayList<>();
    List<SingleMethod> currentMethods = new ArrayList<>();
    int latestGlobalId = 0;
    List<SingleMethod> previousMeth = new ArrayList<>();
    List<SingleMethod> currentMeth = new ArrayList<>();
    List<SingleMethod> methodd = new ArrayList<>();
    AccessPoint ap = new AccessPoint();
    int lowerRevision = ap.firstGenRevision();
    Utility utility = new Utility();
    CommonParameters cp = new CommonParameters();
    String subJectSystem;
    static int prevRev = 1;

    public void methodOrigin() throws CsvValidationException, IOException {

        for (int i = 2; i <= totalRevision; i++) {

            System.out.println("Revision scaning for global id : " + i);
            if (i == 2) {
                computeFirstGlobalMethodId(i);

            } else if (i > 2) {

                computeGlobalMethod(i, i - 1);

            }
        }

    }

    public static String relevantPath(String filePath) {
        // Find the index of "Revision_"
        int revisionIndex = filePath.indexOf("Revision_");

        // Check if "Revision_" is found
        if (revisionIndex != -1) {
            // Extract the substring from "Revision_" to the end of the string
            return filePath.substring(revisionIndex);
        }
        // Return the original string if "Revision_" is not found
        return filePath;
    }

    public void computeFirstGlobalMethodId(int currentRevision) {
        try {

            // currentMethods = ap.getMethods(currentRevision);
            subJectSystem = cp.getSubjectSystem();
            String path = "WorkFolder/" + subJectSystem + "/Methods/GlobalMethods/" + currentRevision + "_methods.txt";
            File outputFile = new File(path);

            // Ensure the directories exist
            outputFile.getParentFile().mkdirs();

            // Open the file in append mode
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile, true));
            currentMethods = ap.getFirstMethods(currentRevision);

            if (outputFile.length() == 0) {
                writer.write(
                        "revision_number,method_name,method_signature,package_name,class_name,src,start_line,end_line,methodId,globalMethodId\n");
            }

            // Map to track signature -> globalMethodId for consolidation
            // Methods with identical signatures will share the same globalMethodId
            java.util.Map<String, Integer> signatureToGlobalId = new java.util.HashMap<>();
            int nextGlobalId = 1;

            for (SingleMethod method : currentMethods) {
                if (method == null || currentMethods == null) {
                    break;
                } else {
                    int revisionNumber = method.revision;
                    String methodName = method.methodName;
                    String signature = method.signature;
                    String packageName = method.packageName;
                    String className = method.className;
                    String filePath = relevantPath(method.filePath);
                    int startLine = method.startLine;
                    int endLine = method.endLine;
                    int methodId = method.methodId;

                    // Check if we've seen this signature before
                    // Use signature as the key for grouping identical methods
                    int globalMethodId;
                    if (signatureToGlobalId.containsKey(signature)) {
                        // Reuse existing globalMethodId for duplicate signature
                        globalMethodId = signatureToGlobalId.get(signature);
                    } else {
                        // New signature - assign new globalMethodId
                        globalMethodId = nextGlobalId++;
                        signatureToGlobalId.put(signature, globalMethodId);
                    }
                    method.globalMethodId = globalMethodId;

                    String data = revisionNumber + " | " + methodName + " | " + signature.replace("\"", "\"\"")
                            + " | " + packageName + " | " + className + " | " + filePath + " | " + startLine
                            + " | " + endLine + " | " + methodId + " | " + globalMethodId;
                    writer.write(data + "\n");

                    // writer.write(String.format("%d | %s | \"%s\" | \"%s\" | %d | %d | %d | %d\n",
                    // revisionNumber, methodName, signature.replace("\"", "\"\""), filePath,
                    // startLine, endLine, methodId, globalMethodId));
                }
            }
            writer.close();
            // Update latestGlobalId to the highest assigned globalMethodId
            latestGlobalId = nextGlobalId - 1;

            currentMethods.clear();
        } catch (IOException e) {
            System.err.println("IOException occurred 1 : " + e.getMessage());
        } catch (NullPointerException e) {
            System.err.println("NullPointerException occurred: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("An unexpected error occurred: " + e.getMessage());
        }

        prevRev = currentRevision;
    }

    // 3 2
    public void computeGlobalMethod(int currentRevision, int previousRevision) {
        try {
            System.out.println("I AM ::::::::::::::::::::::::::::::: " + currentRevision);
            String currentPath = "WorkFolder/" + subJectSystem + "/Methods/GlobalMethods/" + currentRevision
                    + "_methods.txt";
            File outputFile = new File(currentPath);

            // Ensure the directories exist
            outputFile.getParentFile().mkdirs();

            // Open the file in append mode
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile, true))) {
                if (outputFile.length() == 0) {
                    writer.write(
                            "revision_number,method_name,method_signature,package_name,class_name,src,start_line,end_line,methodId,globalMethodId\n");
                }

                List<SingleMethod> previousMethods = ap.getPrevMethods(previousRevision);
                List<SingleMethod> currentMethods = ap.getMethods(currentRevision);

                // System.out.println(" PREVIOUS
                // _________________________________________________________");

                // print(previousMethods);

                // System.out.println(" CURRENT
                // _________________________________________________________");
                // print(currentMethods);

                // Map to track signature -> globalMethodId for consolidation within current
                // revision
                // Methods with identical signatures in the same revision will share the same
                // globalMethodId
                java.util.Map<String, Integer> signatureToGlobalId = new java.util.HashMap<>();

                // Pre-populate map with signatures from previous revision for reference
                for (SingleMethod prevMethod : previousMethods) {
                    if (prevMethod != null && !signatureToGlobalId.containsKey(prevMethod.signature)) {
                        signatureToGlobalId.put(prevMethod.signature, prevMethod.globalMethodId);
                    }
                }

                for (SingleMethod currentMethod : currentMethods) {
                    // System.out.println(currentMethod.revision +" :::: " +currentMethod.methodName
                    // );
                    boolean found = false;
                    if (currentMethod == null) {
                        break;
                    } else {
                        // First, check if this signature already exists in our map (from same revision
                        // or previous)
                        String currentSignature = currentMethod.signature.replace("\"", "");
                        if (signatureToGlobalId.containsKey(currentSignature)) {
                            currentMethod.globalMethodId = signatureToGlobalId.get(currentSignature);
                            found = true;
                        }

                        if (!found) {
                            for (SingleMethod previousMethod : previousMethods) {
                                if (previousMethod == null) {
                                    break;
                                } else {
                                    // System.out.println(currentMethod.methodName + " == " +
                                    // previousMethod.methodName);
                                    String cmethod = currentMethod.methodName.replace("\"", "");
                                    String pmethod = previousMethod.methodName.replace("\"", "");
                                    String cSign = currentMethod.signature.replace("\"", "");
                                    String pSign = previousMethod.signature.replace("\"", "");

                                    if (cmethod.equals(pmethod) || cSign.equals(pSign)) {
                                        currentMethod.globalMethodId = previousMethod.globalMethodId;
                                        found = true;
                                        // Add to map for consolidation with other methods in this revision
                                        signatureToGlobalId.put(currentSignature, currentMethod.globalMethodId);
                                        previousMethods.remove(previousMethod); // Remove the matched method from the
                                                                                // previousMethods list
                                        break; // Exit the loop once a match is found
                                    }
                                }
                            }
                        }
                        if (!found) {

                            String pathaa = currentMethod.filePath.replace("\"", ""); // Remove double quotes from file
                                                                                      // path
                            // System.out.println(pathaa);
                            String cPath = getFileName(pathaa);

                            // Construct full path: prefix + /Revision_N/ + relativePath
                            String fullCurrentPath = prefix + "/Revision_" + currentRevision + "/" + pathaa;

                            // System.out.println(cPath);

                            try {
                                // System.out.println("here is problem");
                                String currentMethodContent = extractMethodContent(fullCurrentPath,
                                        currentMethod.startLine,
                                        currentMethod.endLine);
                                boolean similarityFound = false;
                                // System.out.println("pathaa::::::::::: " +pathaa);
                                for (SingleMethod previousMethod : previousMethods) {
                                    String previousFilePath = previousMethod.filePath.replace("\"", ""); // Remove
                                                                                                         // double
                                                                                                         // quotes from
                                                                                                         // previous
                                                                                                         // file path
                                    String prevPath = getFileName(previousFilePath);
                                    // Construct full path for previous revision
                                    String fullPrevPath = prefix + "/Revision_" + previousRevision + "/"
                                            + previousFilePath;

                                    if (cPath.equals(prevPath)) {

                                        // System.out.println("here is problem = " + ssss + " "
                                        // +previousMethod.startLine + " " +previousMethod.endLine );

                                        // String mm = getFileContentByLineRange(ssss
                                        // ,previousMethod.startLine,previousMethod.endLine );
                                        // System.out.println("here is problem = "+ mm );
                                        String previousMethodContent = extractMethodContent(fullPrevPath,
                                                previousMethod.startLine, previousMethod.endLine);
                                        // System.out.println("here is problem = "+ mm );
                                        boolean flag = Utility.isThisTwoCodeSimilar(currentMethodContent,
                                                previousMethodContent, 75);
                                        // System.out.println("here is problem = " + prevPath + " " +cPath + " " +
                                        // previousMethodContent);
                                        if (flag) {
                                            similarityFound = true;
                                            currentMethod.globalMethodId = previousMethod.globalMethodId;
                                            // Add to map for consolidation
                                            signatureToGlobalId.put(currentMethod.signature.replace("\"", ""),
                                                    currentMethod.globalMethodId);
                                            break;
                                        }
                                    }
                                }

                                if (!similarityFound) {
                                    // Check one more time if another method in this revision has same signature
                                    String cleanSig = currentMethod.signature.replace("\"", "");
                                    if (signatureToGlobalId.containsKey(cleanSig)) {
                                        currentMethod.globalMethodId = signatureToGlobalId.get(cleanSig);
                                    } else {
                                        latestGlobalId++;
                                        currentMethod.globalMethodId = latestGlobalId;
                                        signatureToGlobalId.put(cleanSig, latestGlobalId);
                                        System.out.println("New method introduced : " + latestGlobalId + " "
                                                + currentMethod.methodName);
                                    }
                                }
                            } catch (IOException e) {
                                // If file read fails, still try signature consolidation
                                String cleanSig = currentMethod.signature.replace("\"", "");
                                if (signatureToGlobalId.containsKey(cleanSig)) {
                                    currentMethod.globalMethodId = signatureToGlobalId.get(cleanSig);
                                } else {
                                    latestGlobalId++;
                                    currentMethod.globalMethodId = latestGlobalId;
                                    signatureToGlobalId.put(cleanSig, latestGlobalId);
                                }
                                System.out.println("IOException occurred 2: " + e.getMessage());
                                // Handle the IOException appropriately, e.g., log the error or take corrective
                                // action
                            }
                        }

                    }
                    // System.out.println(currentMethod.revision +" :LAST:: "
                    // +currentMethod.methodName );

                }

                previousMeth.clear();
                previousMeth = currentMethods;

                for (SingleMethod method : currentMethods) {
                    if (method != null) {

                        prevRev = currentRevision;
                        int revisionNumber = method.revision;
                        String methodName = method.methodName;
                        String signature = method.signature;
                        String packageName = method.packageName;
                        String className = method.className;
                        String filePath = relevantPath(method.filePath);
                        int startLine = method.startLine;
                        int endLine = method.endLine;
                        int methodId = method.methodId;
                        int globalMethodId = method.globalMethodId;

                        String data = revisionNumber + " | " + methodName + " | " + signature.replace("\"", "\"\"")
                                + " | " + packageName + " | " + className + " | " + filePath + " | " + startLine
                                + " | " + endLine + " | " + methodId + " | " + globalMethodId;

                        // System.out.println(data);
                        writer.write(data + "\n");

                        // writer.write(String.format("%d | \"%s\" | \"%s\" | \"%s\" | %d | %d | %d |
                        // %d\n",
                        // revisionNumber, methodName, signature.replace("\"", "\"\""), filePath,
                        // startLine, endLine, methodId, globalMethodId));
                    }
                }
                // writer.close() removed — try-with-resources handles closing automatically

            }
        } catch (IOException e) {
            System.err.println("IOException occurred: " + e.getMessage());
        } catch (NullPointerException e) {
            System.err.println("NullPointerException occurred: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("An unexpected error occurred: " + e.getMessage());
        }
    }

    // to get the file name from a src path
    public static String getFileName(String filePath) {
        // Handle both Windows and Unix path separators
        int lastIndex = filePath.lastIndexOf("/");
        if (lastIndex == -1) {
            lastIndex = filePath.lastIndexOf("\\"); // Fallback to backslash
        }
        if (lastIndex == -1) {
            return filePath; // If no separator is found, return the entire path
        } else {
            return filePath.substring(lastIndex + 1); // Return the substring after the last separator
        }
    }

    private String extractMethodContent(String filePath, int startLine, int endLine) throws IOException {
        List<String> allLines = Files.readAllLines(Paths.get(filePath));
        StringBuilder methodContent = new StringBuilder();
        for (int i = startLine - 1; i < endLine; i++) {
            methodContent.append(allLines.get(i)).append("\n");
        }

        return methodContent.toString();
    }

    // private String extractMethodContent(String filePath, int startLine, int
    // endLine) throws IOException {
    // Path path = Paths.get(filePath);
    // StringBuilder builder = new StringBuilder();
    //
    // // Use try-with-resources
    // try (BufferedReader br = Files.newBufferedReader(path,
    // StandardCharsets.UTF_8)) {
    // String line;
    // int current = 1;
    //
    // while ((line = br.readLine()) != null) {
    // if (current >= startLine && current <= endLine) {
    // builder.append(line).append("\n");
    // }
    // if (current > endLine) {
    // break; // Stop reading after endLine
    // }
    // current++;
    // }
    // }
    //
    // return builder.toString();
    // }
    //

    private double calculateSimilarity(String content1, String content2) {
        List<String> list1 = List.of(content1.split("\n"));
        List<String> list2 = List.of(content2.split("\n"));

        Patch<String> patch = DiffUtils.diff(list1, list2);
        int totalLines = Math.max(list1.size(), list2.size());
        int changes = patch.getDeltas().size();

        return 1.0 - (double) changes / totalLines;
    }

    public void print(List<SingleMethod> methods) {
        for (SingleMethod method : methods) {
            if (method == null || methods == null) {
                continue;
            } else {
                System.out.println("Method name: " + method.methodName);
                System.out.println("Signature: " + method.signature);
                System.out.println("Filepath: " + method.filePath);
                System.out.println("Start line: " + method.startLine);
                System.out.println("End line: " + method.endLine);
                System.out.println("MethodId: " + method.methodId);
                System.out.println("GlobalMethodId: " + method.globalMethodId);
            }
        }
    }
}
