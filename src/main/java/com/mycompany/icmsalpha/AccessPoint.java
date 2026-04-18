/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

/**
 *
 * @author Administrator
 */

import difflib.Delta;
import difflib.DiffUtils;
import difflib.Patch;

import java.io.FileNotFoundException;

import java.util.ArrayList;
import java.util.HashSet;

import java.io.File;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

class Clones {
    int cloneId;
    String filePath;
    int startLine;
    int endLine;
    int nLines;
    int globalCloneId;
    int methodId;
    int revision;
    String cloneType;
    int classId;
    int globalClassId;
    int similarity; // Similarity percentage from NiCad (0-100)
}

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
}

class Buggy {
    String fileName;
    String filePath;
    int line;
    int severity;
}

class SingleMethod {
    int revision;
    String methodName;
    String signature;
    String filePath;
    int startLine;
    int endLine;
    int methodId;
    int globalMethodId;
    String packageName = ""; // Java package name
    String className = ""; // Class or file name containing the method

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        SingleMethod that = (SingleMethod) o;
        return revision == that.revision &&
                Objects.equals(methodName, that.methodName) &&
                Objects.equals(signature, that.signature) &&
                Objects.equals(filePath, that.filePath);
    }

    @Override
    public int hashCode() {
        return Objects.hash(revision, methodName, signature, filePath);
    }
}

public class AccessPoint {

    CommonParameters cp = new CommonParameters();
    LoadParam lp = new LoadParam();
    String cloneFolderName;
    String cloneFileName;
    String cloneClassFileName;
    String subJectSystem;
    static int cloneId = 1;
    static boolean first = false;
    static List<Clones> prevClones = new ArrayList<>();
    static Map<String, Integer> globalCloneIdMap = new HashMap<>();
    static Map<String, Integer> globalClassIdMap = new HashMap<>();
    private static int globalClassIdCounter = 1;
    private static int globalCloneId = 1;
    public String WorkFolder;

    public AccessPoint() {
        WorkFolder = cp.WorkFolder;
    }

    public int firstGenRevision() {

        return 3;
    }

    public List<Integer> getBugFixRevision() throws FileNotFoundException, IOException {

        String bugFixRevisionPath = cp.getBugFixRevisionsFilePath();
        List<Integer> bugRevision = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(bugFixRevisionPath));
        String line;
        while ((line = br.readLine()) != null) {
            String data[] = line.split(" ");
            for (String rev : data) {
                bugRevision.add(Integer.parseInt(rev));
            }
        }
        return bugRevision;

    }

    public List<Integer> getBugRevision() throws IOException {
        String bugFixRevisionPath = cp.getBugFixRevisionsFilePath();
        List<Integer> bugRevision = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(bugFixRevisionPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] data = line.split(" ");
                for (String rev : data) {
                    bugRevision.add(Integer.parseInt(rev.trim()) - 1);
                }
            }
        }
        return bugRevision;
    }

    public void saveTheMethods(List<SingleMethod> methods, int revisionNum) throws IOException {
        System.out.println("Saving Revision: " + revisionNum);

        String filePath = WorkFolder + cp.getSubjectSystem() + "/Methods/AllMethods/" + revisionNum + "_methods.txt";
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filePath, true))) {
            for (SingleMethod method : methods) {
                String data = method.revision + " | " + method.methodName + " | "
                        + method.signature.replace("\"", "\"\"") + " | " + method.filePath + " | " + method.startLine
                        + " | " + method.endLine + " | " + method.methodId;
                bw.write(data + "\n");
                // bw.write(String.format("%d | %s | \"%s\" | \"%s\" | %d | %d | %d\n",
                // method.revision, method.methodName, method.signature.replace("\"", "\"\""),
                // method.filePath, method.startLine, method.endLine, method.methodId));
            }
        }
    }

    public List<SingleMethod> getFirstMethods(int revision) throws IOException {
        String filePath = WorkFolder + cp.getSubjectSystem() + "/Methods/AllMethods/" + revision + "_methods.txt";
        String line = "";
        List<SingleMethod> methodSet = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            br.readLine(); // Skip the header line

            while ((line = br.readLine()) != null) {
                // System.out.println(line);
                String[] parts = line.split("\\|");
                int revisionName = Integer.parseInt(parts[0].trim());

                if (revisionName == revision) {
                    SingleMethod method = new SingleMethod();
                    method.revision = revisionName;
                    method.methodName = parts[1].trim();
                    method.signature = parts[2].trim();
                    // New column order: package_name and class_name come before src
                    if (parts.length > 8) {
                        method.packageName = parts[3].trim();
                        method.className = parts[4].trim();
                        method.filePath = parts[5].trim();
                        method.startLine = Integer.parseInt(parts[6].trim());
                        method.endLine = Integer.parseInt(parts[7].trim());
                        method.methodId = Integer.parseInt(parts[8].trim());
                    } else {
                        // Fallback for old format
                        method.filePath = parts[3].trim();
                        method.startLine = Integer.parseInt(parts[4].trim());
                        method.endLine = Integer.parseInt(parts[5].trim());
                        method.methodId = Integer.parseInt(parts[6].trim());
                    }
                    method.globalMethodId = method.methodId;
                    methodSet.add(method); // HashSet will prevent duplicates
                }
            }
        }
        // print(methodSet);
        return methodSet; // Convert Set back to List if required
    }

    public List<SingleMethod> getMethods(int revision) throws IOException {
        String filePath = WorkFolder + cp.getSubjectSystem() + "/Methods/AllMethods/" + revision + "_methods.txt";
        String line = "";
        List<SingleMethod> methodSet = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {

            line = br.readLine(); // skip the first line
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");
                int revisionName = Integer.parseInt(parts[0].trim());

                if (revisionName == revision) {
                    SingleMethod method = new SingleMethod();
                    method.revision = revisionName;
                    method.methodName = parts[1].trim();
                    method.signature = parts[2].trim();
                    // New column order: package_name and class_name come before src
                    if (parts.length > 8) {
                        method.packageName = parts[3].trim();
                        method.className = parts[4].trim();
                        method.filePath = parts[5].trim();
                        method.startLine = Integer.parseInt(parts[6].trim());
                        method.endLine = Integer.parseInt(parts[7].trim());
                        method.methodId = Integer.parseInt(parts[8].trim());
                    } else {
                        // Fallback for old format
                        method.filePath = parts[3].trim();
                        method.startLine = Integer.parseInt(parts[4].trim());
                        method.endLine = Integer.parseInt(parts[5].trim());
                        method.methodId = Integer.parseInt(parts[6].trim());
                    }

                    methodSet.add(method); // HashSet will prevent duplicates
                }
            }
        }

        return methodSet; // Convert Set back to List if required
    }

    public List<SingleMethod> getPrevMethods(int revision) throws IOException {
        String filePath = WorkFolder + cp.getSubjectSystem() + "/Methods/GlobalMethods/" + revision + "_methods.txt";
        String line = "";
        List<SingleMethod> methodSet = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            br.readLine(); // Skip the header line

            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");
                int revisionName = Integer.parseInt(parts[0].trim());

                if (revisionName == revision) {
                    SingleMethod method = new SingleMethod();
                    method.revision = revisionName;
                    method.methodName = parts[1].trim();
                    method.signature = parts[2].trim();
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

                    // System.out.println("GID : " + Integer.parseInt(parts[7]) + " " +
                    // method.revision);
                    methodSet.add(method); // HashSet will prevent duplicates
                }
            }
        }
        // print(methodSet);
        return methodSet; // Convert Set back to List if required
    }

    // FOR CLONES

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

    public void extractAllClones(int revision, String choice, String granularity) throws Exception {

        String cloneFolderPath = null;
        String cloneFolderName = null;
        String cloneClassFileName = null;

        // Determine the correct folder and file names based on the choice and
        // granularity
        if (choice.equals("Type1")) {
            if (granularity.equals("Function")) {
                cloneFolderName = "_functions-clones";
                cloneClassFileName = "_functions-clones-0.00-classes.xml";
                cloneFolderPath = cp.getT1FClones(revision);
            } else if (granularity.equals("Block")) {
                cloneFolderName = "_blocks-clones";
                cloneClassFileName = "_blocks-clones-0.00-classes.xml";
                cloneFolderPath = cp.getT1BClones(revision);
            }
        } else if (choice.equals("Type2")) {
            if (granularity.equals("Block")) {
                cloneFolderName = "_blocks-blind-clones";
                cloneClassFileName = "_blocks-blind-clones-0.00-classes.xml";
                cloneFolderPath = cp.getT2BClones(revision);
            } else if (granularity.equals("Function")) {
                cloneFolderName = "_functions-blind-clones";
                cloneClassFileName = "_functions-blind-clones-0.00-classes.xml";
                cloneFolderPath = cp.getT2FClones(revision);
            }
        } else if (choice.equals("Type3")) {
            if (granularity.equals("Function")) {
                cloneFolderName = "_functions-blind-clones";
                cloneClassFileName = "_functions-blind-clones-0.30-classes.xml";
                cloneFolderPath = cp.getT3FClones(revision);
            } else if (granularity.equals("Block")) {
                cloneFolderName = "_blocks-blind-clones";
                cloneClassFileName = "_blocks-blind-clones-0.30-classes.xml";
                cloneFolderPath = cp.getT3BClones(revision);
            }
        }

        String xmlFilePath = cp.getCloneDir() + "/Revision_" + revision + cloneFolderName + "/Revision_" + revision
                + cloneClassFileName;

        try {
            File inputFile = new File(xmlFilePath);
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(inputFile);
            doc.getDocumentElement().normalize();

            NodeList classList = doc.getElementsByTagName("class");
            // Create a FileWriter to write the output to a text file
            FileWriter writer = new FileWriter(cloneFolderPath);
            // writer.write("Revision | filePath | changeType | StartLine | EndLine |
            // cloneId | classId\n");
            writer.write(
                    "Revision | filePath | changeType | StartLine | EndLine | Nlines | cloneId | classId | Similarity\n");

            for (int i = 0; i < classList.getLength(); i++) {
                Node classNode = classList.item(i);
                if (classNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element classElement = (Element) classNode;
                    String classID = classElement.getAttribute("classid");
                    String nlines = classElement.getAttribute("nlines");
                    String similarity = classElement.getAttribute("similarity");
                    // Default to 100 if similarity attribute is missing (Type 1/2 exact clones)
                    if (similarity == null || similarity.isEmpty()) {
                        similarity = "100";
                    }

                    NodeList sourceList = classElement.getElementsByTagName("source");
                    for (int j = 0; j < sourceList.getLength(); j++) {

                        Node sourceNode = sourceList.item(j);
                        if (sourceNode.getNodeType() == Node.ELEMENT_NODE) {
                            Element sourceElement = (Element) sourceNode;
                            String ffilePath = sourceElement.getAttribute("file");
                            String filePath = relevantPath(ffilePath);
                            String startLine = sourceElement.getAttribute("startline");
                            String endLine = sourceElement.getAttribute("endline");
                            String changeType = "A";

                            // Create a unique key for the clone
                            if (revision == 2) { // take the first clone
                                changeType = "A";
                            } else if (revision > 2) {

                                changeType = determineChangeType(filePath, Integer.parseInt(startLine),
                                        Integer.parseInt(endLine), revision, revision - 1);
                            }
                            // System.out.println( filePath + " " + startLine+ " " + endLine + " | " +
                            // changeType);
                            // Write the clone details to the text file
                            // writer.write(revision + " | " + filePath + " | " + changeType + " | " +
                            // startLine + " | " + endLine + " | " + cloneId + " | " + classID + "\n");

                            writer.write(revision + " | " + filePath + " | " + changeType + " | " + startLine + " | "
                                    + endLine + " | " + nlines + " | " + cloneId + " | " + classID + " | " + similarity
                                    + "\n");
                            // print("CloneID : " + cloneId);
                            cloneId++;
                        }
                    }

                }
            }

            // Close the FileWriter
            writer.close();

            System.out.println("Clone details saved to: " + cloneFolderPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // class ID with global class Id
    public void extractAllClonesWithClassID(int revision, String choice, String granularity) throws Exception {

    }

    public static String extractCodeFragment(String filePath, int startLine, int endLine) throws IOException {
        StringBuilder codeFragment = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            int currentLine = 1;

            while ((line = reader.readLine()) != null) {
                if (currentLine >= startLine && currentLine <= endLine) {
                    codeFragment.append(line).append("\n");
                }
                if (currentLine > endLine) {
                    break;
                }
                currentLine++;
            }
        }

        return codeFragment.toString();
    }

    private String determineChangeType(String filePath, int startLine, int endLine, int currentRevision,
            int prevRevision) {
        try {
            // Load current code fragment from file
            String path = getPath(filePath);
            String cPath = cp.getCloneDir() + "/Revision_" + currentRevision + "/" + path;
            String pPath = cp.getCloneDir() + "/Revision_" + prevRevision + "/" + path;

            List<String> currentLines = loadCodeFragment(cPath, startLine, endLine);

            // Load previous code fragment from previous revision
            List<String> previousLines = loadCodeFragment(pPath, startLine, endLine);

            //

            if (previousLines == null || previousLines.isEmpty()) {
                System.out.println("ADDED");
                return "A"; // Added
            }

            // Compare code fragments using diff
            Patch<String> patch = DiffUtils.diff(previousLines, currentLines);

            // Analyze the patch to determine change type
            for (Delta<String> delta : patch.getDeltas()) {
                if (delta.getType() == Delta.TYPE.CHANGE) {
                    System.out.println("MODIFIED");
                    return "M"; // Modified
                }
            }

            // If no change delta found, consider it unchanged
            return "U"; // Unchanged

        } catch (Exception e) {
            e.printStackTrace();
            return "U"; // Default to Unchanged in case of error
        }
    }

    /**
     * Normalize file path for comparison (removes all Revision_* prefixes)
     * Example: "Revision_300/Revision_300/asp.c" -> "asp.c"
     */
    private static String getPath(String filePath) {
        // Handle both / and \ separators
        String normalized = filePath.replace("\\", "/");

        // Remove all Revision_* prefixes from the path
        String result = normalized.replaceAll("Revision_\\d+/", "");

        // If result is empty, return original
        if (result.isEmpty()) {
            return normalized;
        }

        return result;
    }

    /**
     * Load code fragment from file.
     *
     * @param filePath  File path of the code fragment.
     * @param startLine Starting line number of the code fragment.
     * @param endLine   Ending line number of the code fragment.
     * @return List of strings representing lines of code fragment.
     */
    private List<String> loadCodeFragment(String filePath, int startLine, int endLine) {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            int currentLine = 1;
            while ((line = reader.readLine()) != null) {
                if (currentLine >= startLine && currentLine <= endLine) {
                    lines.add(line);
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
        return lines;
    }

    // public void extractAllClones(int revision , String choice , String
    // granularity) throws Exception {
    //
    // String cloneFolderPath = null;
    // if(choice.equals("Type1")){ // type 1 cloneRevision_2_functions-clones
    // if(granularity.equals("Function")){
    // cloneFolderName = "_functions-clones";
    // //cloneFileName = "_blocks-clones-0.00.xml";
    // cloneClassFileName = "_functions-clones-0.00-classes.xml";
    // cloneFolderPath = cp.getT1FClones(revision);
    //
    // }
    // else if(granularity.equals("Block")){
    // cloneFolderName = "_blocks-clones";
    // //cloneFileName = "_blocks-clones-0.00.xml";
    // //Revision_2_blocks-clones-0.00-classes.xml
    // cloneClassFileName = "_blocks-clones-0.00-classes.xml";
    // cloneFolderPath = cp.getT1BClones(revision);
    // }
    // }
    // else if(choice.equals("Type2")){ // type 2 _blocks-blind-clones
    // Revision_2_blocks-blind-clones-0.00-classes.xml
    // if(granularity.equals("Block")){
    // cloneFolderName = "_blocks-blind-clones";
    // //cloneFileName = "_blocks-clones-0.00.xml";
    // cloneClassFileName = "_blocks-blind-clones-0.00-classes.xml";
    // cloneFolderPath = cp.getT2BClones(revision);
    // }
    // else if(granularity.equals("Function")){
    // cloneFolderName = "_functions-blind-clones";
    // //cloneFileName = "_blocks-clones-0.00.xml";
    // //Revision_2_blocks-clones-0.00-classes.xml
    // cloneClassFileName = "_functions-blind-clones-0.00-classes.xml";
    // cloneFolderPath = cp.getT2FClones(revision);
    // }
    // }
    //
    //
    //
    //
    // //D:\Thesis\NiCad-6.2\systems\ctags\Revision_2_blocks-clones
    // //D:\Thesis\NiCad-6.2\systems\ctags\Revision_2_blocks-clones\Revision_2_blocks-clones-0.00-classes.xml
    // String xmlFilePath =
    // cp.getCloneDir()+"\\Revision_"+revision+cloneFolderName+"\\Revision_"+revision
    // +cloneClassFileName;
    //
    // try {
    // // Path to your XML file
    // File inputFile = new File(xmlFilePath);
    // // Create a DocumentBuilderFactory and set it up
    // DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
    // DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
    //
    // // Parse the XML file and create a Document object
    // Document doc = dBuilder.parse(inputFile);
    // doc.getDocumentElement().normalize();
    //
    // // Get all <class> elements
    // NodeList classList = doc.getElementsByTagName("class");
    //
    // // HashMap to store unique clones and their assigned IDs
    // Map<String, Integer> globalCloneIdMap = new HashMap<>();
    // int globalCloneId = 1;
    //
    // // Create a FileWriter to write the output to a text file
    //
    //
    // FileWriter writer = new FileWriter(cloneFolderPath);
    // writer.write("CloneId | path | startLine | endLine\n");
    //
    // // Iterate through all <class> elements
    // for (int i = 0; i < classList.getLength(); i++) {
    // Node classNode = classList.item(i);
    // if (classNode.getNodeType() == Node.ELEMENT_NODE) {
    // Element classElement = (Element) classNode;
    // String nlines = classElement.getAttribute("nlines");
    // // Get all <source> elements within the current <class>
    //
    // NodeList sourceList = classElement.getElementsByTagName("source");
    // int nLines = Integer.parseInt(nlines);
    // for (int j = 0; j < sourceList.getLength(); j++) {
    // Node sourceNode = sourceList.item(j);
    // if (sourceNode.getNodeType() == Node.ELEMENT_NODE) {
    // Element sourceElement = (Element) sourceNode;
    // String ffilePath = sourceElement.getAttribute("file");
    // String filePath = relevantPath(ffilePath);
    // String startLine = sourceElement.getAttribute("startline");
    // String endLine = sourceElement.getAttribute("endline");
    // boolean flag = isMicroClone(Integer.parseInt(startLine),
    // Integer.parseInt(endLine));
    // if(!flag){
    // System.out.println("Not taken");
    // continue;
    // }else
    //
    // {
    // // Create a unique key for the clone
    // String cloneKey = filePath + ":" + startLine + "-" + endLine;
    //
    // // Check if this clone is already assigned a global ID
    // if (!globalCloneIdMap.containsKey(cloneKey)) {
    // globalCloneIdMap.put(cloneKey, globalCloneId++);
    // }
    //
    // // Assign the global clone ID to this source
    // int assignedId = globalCloneIdMap.get(cloneKey);
    //
    // // Write the clone details to the text file
    // writer.write(cloneId + " | " + filePath + " | " + startLine + " | " + endLine
    // + "\n");
    // cloneId++;
    // }
    // }
    // }
    //
    //
    // }
    // }
    // // Close the FileWriter
    // writer.close();
    //
    // System.out.println("Clone details saved to : " +revision+"_clones.txt");
    //
    //
    // } catch (Exception e) {
    // e.printStackTrace();
    // }
    //
    // }
    //
    public boolean isMicroClone1(int line, String gran) {
        boolean flag = false;
        if (gran.equals("Function")) {

            if (line >= 3 && line <= 4) {
                flag = true;
            }
        }

        if (gran.equals("Block")) {

            if (line > 3 && line < 5) {
                flag = true;
            }
        }

        return flag;
    }

    public boolean isMicroClone(int line, String gran) {
        boolean flag = false;
        if (gran.equals("Function")) {

            if (line >= 3 && line <= 6) {
                flag = true;
            }
        }

        if (gran.equals("Block")) {

            if (line >= 3 && line <= 5) {
                flag = true;
            }
        }

        return flag;
    }

    public boolean isRegularClone(int line, String gran) {
        boolean flag = false;
        if (gran.equals("Function")) {
            if (line > 6) {
                flag = true;
            }
        }
        if (gran.equals("Block")) {

            if (line > 6) {
                flag = true;
            }
        }

        return flag;
    }

    public void getMeAllClones(int prevRevision, String choice, String granularity) throws IOException {
        String prevCloneFile;
        String globalCloneFile;
        System.out.println("Called");

        prevCloneFile = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + choice + "/" + granularity
                + "/AllClones/" + prevRevision + "_clones.txt";
        print(prevCloneFile);

        List<Clones> cloneSet = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(prevCloneFile))) {
            br.readLine(); // skip the first line
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);

            }
        }

    }

    public void setFirstGlobalClones(int prevRevision, String choice, String granularity) throws IOException {
        String prevCloneFile;
        String globalCloneFile;
        System.out.println(prevRevision + " " + choice + " " + granularity);

        prevCloneFile = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + choice + "/" + granularity
                + "/AllClones/" + prevRevision + "_clones.txt";
        globalCloneFile = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + choice + "/" + granularity
                + "/GlobalClones/" + prevRevision + "_clones.txt";
        print(prevCloneFile);

        List<Clones> cloneSet = new ArrayList<>();

        // Ensure the directories exist
        File outputFile = new File(globalCloneFile);
        outputFile.getParentFile().mkdirs();

        try (BufferedReader br = new BufferedReader(new FileReader(prevCloneFile));
                BufferedWriter bw = new BufferedWriter(new FileWriter(globalCloneFile, true))) {
            br.readLine(); // skip the first line

            if (outputFile.length() == 0) {

                // bw.write("Revision | filePath | Path | changeType | StartLine | EndLine |
                // cloneId | classId | globalCloneId \n");
                bw.write(
                        "Revision | filePath | changeType | StartLine | EndLine | Nlines | cloneId | classId | globalCloneId | Similarity\n");

            }

            String line;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");
                System.out.println(line.length());

                Clones clone = new Clones();
                // Revision | filePath | changeType | StartLine | EndLine | cloneId | classId |
                // globalCloneId | globalClassid
                clone.revision = Integer.parseInt(parts[0].trim());
                clone.filePath = parts[1].trim();
                clone.cloneType = parts[2].trim();
                clone.startLine = Integer.parseInt(parts[3].trim());
                clone.endLine = Integer.parseInt(parts[4].trim()); // Corrected index for endLine
                // System.out.println(clone.revision + " | "+clone.filePath + " |
                // "+clone.cloneType + " | "+clone.startLine + " | "+clone.endLine );
                clone.nLines = Integer.parseInt(parts[5].trim());
                clone.cloneId = Integer.parseInt(parts[6].trim());
                clone.classId = Integer.parseInt(parts[7].trim());
                // Parse similarity from AllClones (index 8)
                clone.similarity = parts.length > 8 ? Integer.parseInt(parts[8].trim()) : 100;
                clone.globalCloneId = clone.cloneId;
                clone.globalClassId = clone.classId;
                System.out.println(clone.revision + " | " + clone.filePath + " | " + clone.cloneType + " | "
                        + clone.startLine + " | " + clone.endLine);

                cloneSet.add(clone);
                // System.out.println(clone.cloneId + " " + clone.filePath + " " +
                // clone.startLine + " " + clone.endLine);
                String data = clone.revision + " | " + clone.filePath + " | " + clone.cloneType + " | "
                        + clone.startLine + " | " + clone.endLine + " | " + clone.nLines + " | " + clone.cloneId + " | "
                        + clone.classId + " | " + clone.globalCloneId + " | " + clone.similarity + "\n";

                bw.write(data);

            }
        }
    }

    public List<Clones> getFirstClones(int revision, String choice, String granularity) throws IOException {
        String path = null;

        path = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + choice + "/" + granularity + "/GlobalClones/"
                + revision + "_clones.txt";

        List<Clones> cloneSet = new ArrayList<>();

        // Ensure the directories exist
        File outputFile = new File(path);
        outputFile.getParentFile().mkdirs();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String header = br.readLine(); // Read the header line
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");

                Clones clone = new Clones();
                // Revision | filePath | changeType | StartLine | EndLine | cloneId | classId |
                // globalCloneId
                clone.revision = Integer.parseInt(parts[0].trim());
                clone.filePath = parts[1].trim();
                clone.cloneType = parts[2].trim();

                clone.startLine = Integer.parseInt(parts[3].trim());
                clone.endLine = Integer.parseInt(parts[4].trim()); // Corrected index for endLine
                clone.nLines = Integer.parseInt(parts[5].trim()); // Corrected index for endLine

                clone.cloneId = Integer.parseInt(parts[6].trim());
                clone.classId = Integer.parseInt(parts[7].trim());

                clone.globalCloneId = Integer.parseInt(parts[8].trim());
                // Parse similarity (index 9) - default to 100 if not present
                clone.similarity = parts.length > 9 ? Integer.parseInt(parts[9].trim()) : 100;

                cloneSet.add(clone);
                // System.out.println(clone.cloneId + " " + clone.filePath + " " +
                // clone.startLine + " " + clone.endLine);

            }
        }
        return cloneSet;
    }

    // save global final clone with global class id
    public void setFirstGlobalClassId(int prevRevision, String choice, String granularity)
            throws FileNotFoundException, IOException {
        String prevCloneFile, globalCloneFile;

        prevCloneFile = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + choice + "/" + granularity
                + "/AllClones/" + prevRevision + "_clones.txt";
        globalCloneFile = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + choice + "/" + granularity
                + "/GlobalFinalClones/" + prevRevision + "_clones.txt";

        List<Clones> cloneSet = new ArrayList<>();

        // Ensure the directories exist
        File outputFile = new File(globalCloneFile);
        outputFile.getParentFile().mkdirs();

        try (BufferedReader br = new BufferedReader(new FileReader(prevCloneFile));
                BufferedWriter bw = new BufferedWriter(new FileWriter(globalCloneFile, true))) {
            br.readLine(); // skip the first line

            if (outputFile.length() == 0) {
                bw.write(
                        "Revision | filePath | Path | changeType | StartLine | EndLine | cloneId | classId |  globalCloneId | globalClassId \n");
            }

            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");

                Clones clone = new Clones();
                // Revision | filePath | changeType | StartLine | EndLine | cloneId | classId |
                // globalCloneId | globalClassid
                clone.revision = Integer.parseInt(parts[0].trim());
                clone.filePath = parts[1].trim();
                clone.cloneType = parts[2].trim();

                clone.startLine = Integer.parseInt(parts[3].trim());
                clone.endLine = Integer.parseInt(parts[4].trim()); // Corrected index for endLine
                clone.cloneId = Integer.parseInt(parts[5].trim());
                clone.classId = Integer.parseInt(parts[6].trim());

                clone.globalCloneId = clone.cloneId;
                clone.globalClassId = clone.classId;
                cloneSet.add(clone);
                // System.out.println(clone.cloneId + " " + clone.filePath + " " +
                // clone.startLine + " " + clone.endLine);
                String data = clone.revision + " | " + clone.filePath + " | " + clone.cloneType + " | "
                        + clone.startLine + " | " + clone.endLine + " | " + clone.cloneId + " | " + clone.classId
                        + " | " + clone.globalCloneId + " | " + clone.globalClassId + "\n";
                bw.write(data);
            }
        }
    }

    public List<Clones> getFirstGlobalClassId(int revision, String choice, String granularity) throws IOException {
        String path = null;

        path = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + choice + "/" + granularity
                + "/GlobalFinalClones/" + revision + "_clones.txt";

        List<Clones> cloneSet = new ArrayList<>();

        // Ensure the directories exist
        File outputFile = new File(path);
        outputFile.getParentFile().mkdirs();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String header = br.readLine(); // Read the header line
            String line;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");
                // System.out.println(line);
                Clones clone = new Clones();
                // Revision | filePath | changeType | StartLine | EndLine | cloneId | classId |
                // globalCloneId
                clone.revision = Integer.parseInt(parts[0].trim());
                clone.filePath = parts[1].trim();
                clone.cloneType = parts[2].trim();

                clone.startLine = Integer.parseInt(parts[3].trim());
                clone.endLine = Integer.parseInt(parts[4].trim()); // Corrected index for endLine
                clone.cloneId = Integer.parseInt(parts[5].trim());
                clone.classId = Integer.parseInt(parts[6].trim());

                clone.globalCloneId = Integer.parseInt(parts[7].trim());
                clone.globalClassId = Integer.parseInt(parts[8].trim());
                cloneSet.add(clone);
                // System.out.println(clone.cloneId + " " + clone.filePath + " " +
                // clone.startLine + " " + clone.endLine + " " );

            }
        }
        return cloneSet;
    }

    public List<Clones> getGlobalClones(int revision, String choice, String granularity) throws IOException {
        String path;

        path = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + choice + "/" + granularity + "/GlobalClones/"
                + revision + "_clones.txt";

        List<Clones> cloneSet = new ArrayList<>();

        // Ensure the directories exist
        File outputFile = new File(path);
        outputFile.getParentFile().mkdirs();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String header = br.readLine(); // Read the header line
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");
                // System.out.println(line);
                Clones clone = new Clones();
                // Revision | filePath | changeType | StartLine | EndLine | cloneId | classId |
                // globalCloneId
                clone.revision = Integer.parseInt(parts[0].trim());
                clone.filePath = parts[1].trim();
                clone.cloneType = parts[2].trim();

                clone.startLine = Integer.parseInt(parts[3].trim());
                clone.endLine = Integer.parseInt(parts[4].trim()); // Corrected index for endLine

                clone.nLines = Integer.parseInt(parts[5].trim());
                clone.cloneId = Integer.parseInt(parts[6].trim());
                clone.classId = Integer.parseInt(parts[7].trim());
                clone.globalCloneId = Integer.parseInt(parts[8].trim());
                // Parse similarity (index 9) - default to 100 if column missing
                clone.similarity = parts.length > 9 ? Integer.parseInt(parts[9].trim()) : 100;

                // this is for fun
                /*
                 * clone.cloneId = Integer.parseInt(parts[5].trim());
                 * clone.classId = Integer.parseInt(parts[6].trim());
                 * clone.globalCloneId = Integer.parseInt(parts[7].trim());
                 */

                cloneSet.add(clone);
                // System.out.println(clone.cloneId + " " + clone.filePath + " " +
                // clone.startLine + " " + clone.endLine);
            }
        }
        return cloneSet;
    }

    public List<Clones> getClones(int revision, String choice, String granularity) throws IOException {

        String path;

        path = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/" + choice + "/" + granularity + "/AllClones/"
                + revision + "_clones.txt";

        List<Clones> cloneSet = new ArrayList<>();

        // Ensure the directories exist
        File outputFile = new File(path);
        outputFile.getParentFile().mkdirs();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String header = br.readLine(); // Read the header line
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");

                Clones clone = new Clones();
                // Revision | filePath | changeType | StartLine | EndLine | cloneId | classId |
                // globalCloneId
                clone.revision = Integer.parseInt(parts[0].trim());
                clone.filePath = parts[1].trim();
                clone.cloneType = parts[2].trim();

                clone.startLine = Integer.parseInt(parts[3].trim());
                clone.endLine = Integer.parseInt(parts[4].trim()); // Corrected index for endLine
                clone.nLines = Integer.parseInt(parts[5].trim());

                clone.cloneId = Integer.parseInt(parts[6].trim());
                clone.classId = Integer.parseInt(parts[7].trim());
                // Parse similarity (index 8) - default to 100 if not present
                clone.similarity = parts.length > 8 ? Integer.parseInt(parts[8].trim()) : 100;
                cloneSet.add(clone);
                // System.out.println(clone.cloneId + " " + clone.filePath + " " +
                // clone.startLine + " " + clone.endLine);

            }
        }
        return cloneSet;
    }

    public List<SingleMethod> getFinalMethods(int revision) throws FileNotFoundException, IOException {

        String filePath;

        filePath = "WorkFolder/" + cp.getSubjectSystem() + "/Methods/GlobalMethods/" + revision + "_methods.txt";

        String line = "";
        List<SingleMethod> methodSet = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            br.readLine(); // Skip the header line

            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");
                int revisionName = Integer.parseInt(parts[0].trim());

                if (revisionName == revision) {
                    SingleMethod method = new SingleMethod();
                    method.revision = revisionName;
                    method.methodName = parts[1].trim();
                    method.signature = parts[2].trim();
                    method.filePath = parts[3].trim();
                    method.startLine = Integer.parseInt(parts[4].trim());
                    method.endLine = Integer.parseInt(parts[5].trim());
                    method.methodId = Integer.parseInt(parts[6].trim());
                    method.globalMethodId = Integer.parseInt(parts[7].trim());

                    // System.out.println("GID : " + Integer.parseInt(parts[7]) + " " +
                    // method.revision);
                    methodSet.add(method); // HashSet will prevent duplicates
                }
            }
        }
        // print(methodSet);
        return methodSet; // Convert Set back to List if required
    }

    public List<Clones> getFinalClones(int revision) throws IOException {
        String path;

        path = "WorkFolder/" + cp.getSubjectSystem() + "/Clones/GlobalClones/" + revision + "_clones.txt";

        List<Clones> cloneSet = new ArrayList<>();

        // Ensure the directories exist
        File outputFile = new File(path);
        outputFile.getParentFile().mkdirs();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String header = br.readLine(); // Read the header line
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");

                Clones clone = new Clones();
                clone.cloneId = Integer.parseInt(parts[1].trim());
                clone.filePath = parts[2].trim();
                clone.startLine = Integer.parseInt(parts[3].trim());
                clone.endLine = Integer.parseInt(parts[4].trim()); // Corrected index for endLine
                clone.globalCloneId = Integer.parseInt(parts[5].trim());
                cloneSet.add(clone);
                // System.out.println(clone.cloneId + " " + clone.filePath + " " +
                // clone.startLine + " " + clone.endLine);

            }
        }
        return cloneSet;
    }

    public void print(List<SingleMethod> methods) {
        for (SingleMethod method : methods) {

            System.out.println("Revision name: " + method.revision);
            System.out.println("Method name: " + method.methodName);
            System.out.println("Signature: " + method.signature);
            System.out.println("Filepath: " + method.filePath);
            System.out.println("Start line: " + method.startLine);
            System.out.println("End line: " + method.endLine);
            System.out.println("Start line: " + method.startLine);
            System.out.println("methodId: " + method.methodId);
            System.out.println("globalMethodId: " + method.globalMethodId);

        }
    }

    public String getCode(int startLine, int endLine, File file, int revision) {
        return null;
    }

    public List<SingleMethod> getAllMethodOfRevision(int revision) {

        return null;
    }

    // *********************************************BUGS
    // ******************************************
    // i have all the bug revision and bug fix revision
    // now i want to parse all the changes for both bug revision and bug fix
    // revision

    // ********************************************************************************************

    public void getAllBugRevisionInfo() throws IOException {
        System.out.println("Start Bug revision parsing");
        getBugRevisionInfo();
        System.out.println("End Bug revision parsing");
    }

    // save all the bug revision and bug fix revision
    public void getBugRevisionInfo() throws IOException {
        // String bugRevisionFilePath = cp.getBugRevisionsFilePath();
        // List<String> bugRevision = getBugRevision(bugRevisionFilePath);
        String bugFixRevisionFilePath = cp.getBugFixRevisionsFilePath();
        List<String> bugFixRevision = getBugRevision(bugFixRevisionFilePath);

        List<Integer> allBugsRevision = new ArrayList<>();
        for (String revision : bugFixRevision) {
            allBugsRevision.add(Integer.parseInt(revision));
        }
        // for(String revision : bugRevision){
        // allBugsRevision.add(Integer.parseInt(revision));
        // }
        // // Sort the list

        getAllChangesForBugRevision(allBugsRevision);

    }

    // save all the buggy code changes like revision number
    public void getAllChangesForBugRevision(List<Integer> allBugRevision) throws FileNotFoundException, IOException {
        List<Change> changes = new ArrayList<>();
        String line;
        String changeLogFile = cp.getLogFile();
        BufferedReader br = new BufferedReader(new FileReader(changeLogFile));
        while ((line = br.readLine()) != null) {
            String[] data = line.split("\\|");
            // int rev = Integer.parseInt(data[0].trim());
            if (data.length == 6) {
                Change change = new Change();
                change.revision = Integer.parseInt(data[0].trim());
                change.filePath = data[1].trim();
                change.fileName = data[2].trim();
                change.changeType = data[3].trim();
                change.startLine = Integer.parseInt(data[4].trim());
                change.endLine = Integer.parseInt(data[5].trim());

                changes.add(change);
            }
        }

        List<Change> BugFixChanges = new ArrayList<>();
        for (int bugFixRevision : allBugRevision) {
            for (Change change : changes) {
                if (bugFixRevision == change.revision) {
                    BugFixChanges.add(change);
                }
            }
        }

        saveBugFixInfo(BugFixChanges);
    }

    public void saveBugFixInfo(List<Change> changes) throws IOException {

        String bugFixInfo = cp.getBugFixInfo();
        BufferedWriter br = new BufferedWriter(new FileWriter(bugFixInfo));
        String ss;
        for (Change change : changes) {
            ss = change.revision + " | " + change.filePath + " | " + change.fileName + " | " + change.changeType + " | "
                    + change.startLine + " | " + change.endLine + "\n";
            br.write(ss);
        }
        br.close();
    }

    public List<String> getBugRevision(String filePath) throws FileNotFoundException, IOException {

        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        List<String> revisions = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            String data[] = line.split(" ");
            int k = 0;
            for (int i = 0; i < data.length; i++) {

                revisions.add(data[i]);
            }
            break;
        }
        return revisions;
    }

    // recall all the changes in method
    public List<Changes> getChanges() throws IOException {
        List<Changes> changes = new ArrayList<>();
        String changeRevisionFilePath = cp.getAllChangeRevisionInfoPath();
        BufferedReader br = new BufferedReader(new FileReader(changeRevisionFilePath));
        String line;

        while ((line = br.readLine()) != null) {
            String data[] = line.split("\\|");

            if (data.length >= 6) { // Ensure there are enough elements
                Changes change = new Changes();
                change.revision = Integer.parseInt(data[0].trim());
                change.filePath = data[1].trim();
                change.fileName = data[2].trim();
                change.changeType = data[3].trim();
                change.startLine = Integer.parseInt(data[4].trim());
                change.endLine = Integer.parseInt(data[5].trim());

                // Only set fixType if it exists
                if (data.length >= 7) {
                    change.fixType = data[6].trim();
                } else {
                    change.fixType = ""; // Default or empty value if fixType is missing
                }

                changes.add(change);
                // System.out.println(line);
            } else {
                System.err.println("Malformed line: " + line);
            }
        }

        br.close();
        return changes;
    }

    // recall all the changes in method
    public List<Changes> getAllChanges() throws IOException {
        List<Changes> changes = new ArrayList<>();
        String changeRevisionFilePath = cp.getAllChangeRevisionInfoPath();
        BufferedReader br = new BufferedReader(new FileReader(changeRevisionFilePath));
        String line;
        while ((line = br.readLine()) != null) {
            String data[] = line.split("\\|");
            if (data.length == 7) {
                Changes change = new Changes();
                change.revision = Integer.parseInt(data[0].trim());
                change.filePath = data[1].trim();
                change.fileName = data[2].trim();
                change.changeType = data[3].trim();
                change.startLine = Integer.parseInt(data[4].trim());
                change.endLine = Integer.parseInt(data[5].trim());
                change.fixType = data[6].trim();
                changes.add(change);
            }
        }
        return changes;
    }

    // recall all the changes in method
    public List<Buggy> getBugChanges(int revision) throws IOException {
        List<Buggy> bugChanges = new ArrayList<>();
        String changesPath = cp.getBugAnalyzerSeverityPath(revision);
        BufferedReader br = new BufferedReader(new FileReader(changesPath));
        String line;
        while ((line = br.readLine()) != null) {
            String data[] = line.split("\\|");

            if (data.length == 4) {
                Buggy change = new Buggy();

                change.fileName = data[0].trim();
                change.filePath = data[1].trim();
                change.line = Integer.parseInt(data[2].trim());
                change.severity = Integer.parseInt(data[3].trim());

                bugChanges.add(change);

            }
        }
        return bugChanges;
    }

    /*
     * 
     * get total line of code and total loc of micro clone
     * 
     */
    List<Loc> getMicroCloneLoc(String cloneType, String granularity) throws FileNotFoundException, IOException {
        List<Loc> microCloneLoc = new ArrayList<>();
        String path = cp.getCloneLOCPath(cloneType, granularity);
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        Loc loc;
        while ((line = br.readLine()) != null) {
            String[] data = line.trim().split("\\|");
            loc = new Loc();
            loc.revision = Integer.parseInt(data[0].trim());
            loc.lineCount = Integer.parseInt(data[1].trim());
            microCloneLoc.add(loc);
        }
        return microCloneLoc;
    }

    // get revision Loc
    List<Loc> getRevisioLoc() throws FileNotFoundException, IOException {
        List<Loc> revisionLoc = new ArrayList<>();
        String path = cp.getRevisionLoc();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        Loc loc;
        while ((line = br.readLine()) != null) {
            String[] data = line.trim().split("\\|");
            loc = new Loc();
            loc.revision = Integer.parseInt(data[0].trim());
            loc.lineCount = Integer.parseInt(data[1].trim());
            revisionLoc.add(loc);
        }
        return revisionLoc;
    }

    int getLineCount(List<Loc> LocList) {
        int count = 0;
        for (Loc loc : LocList) {
            count += loc.lineCount;
        }
        return count;
    }

    void getBuggyChanges() throws FileNotFoundException, IOException {
        List<Changes> changes = new ArrayList<>();
        String changesPath = "ChangeLog.txt";
        BufferedReader br = new BufferedReader(new FileReader(changesPath));
        String line;
        while ((line = br.readLine()) != null) {
            String data[] = line.split("\\|");
            if (data.length == 7) {
                Changes change = new Changes();
                change.revision = Integer.parseInt(data[0].trim());
                change.filePath = data[1].trim();
                change.fileName = data[2].trim();
                change.changeType = data[3].trim();
                change.startLine = Integer.parseInt(data[4].trim());
                change.endLine = Integer.parseInt(data[5].trim());
                change.fixType = data[6].trim();
                changes.add(change);
            }
        }

    }

    /* Temporal analysis workdone */

    public List<CloneMatcherOb> getCloneMatcher(String filePath) throws IOException {
        List<CloneMatcherOb> cloneMatches = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\|");

                // Ensure the line contains at least 7 parts to avoid parsing errors
                if (parts.length >= 8) {
                    CloneMatcherOb cloneMatch = new CloneMatcherOb();

                    // Parse previous revision info
                    cloneMatch.index = Integer.parseInt(parts[0].trim());
                    cloneMatch.prevRevision = Integer.parseInt(parts[1].trim());
                    cloneMatch.prevFilePath = parts[2].trim();
                    String[] prevLines = parts[3].trim().split(",");
                    cloneMatch.prevStartLine = Integer.parseInt(prevLines[0].trim());
                    cloneMatch.prevEndLine = (prevLines.length > 1) ? Integer.parseInt(prevLines[1].trim())
                            : cloneMatch.prevStartLine;

                    // Parse change type
                    cloneMatch.changeType = parts[4].trim();

                    // Parse current revision info
                    cloneMatch.currRevision = Integer.parseInt(parts[5].trim());
                    cloneMatch.currFilePath = parts[6].trim();
                    String[] currLines = parts[7].trim().split(",");
                    cloneMatch.currStartLine = Integer.parseInt(currLines[0].trim());
                    cloneMatch.currEndLine = (currLines.length > 1) ? Integer.parseInt(currLines[1].trim())
                            : cloneMatch.currStartLine;

                    // Add the parsed object to the list
                    cloneMatches.add(cloneMatch);
                } else {
                    System.err.println("Malformed line: " + line);
                }
            }
        }

        return cloneMatches;
    }

    public boolean isValid(String path) {

        File file = new File(path);
        if (!file.exists()) {
            return false;
        }
        return true;

    }

    List<SVNChanges> getSVNChanges() throws IOException {

        print("inside the getChanges");
        String path = cp.getDirectory() + "/commit_logs.txt";
        List<SVNChanges> changes = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        boolean aa = false;
        while ((line = br.readLine()) != null) {
            if (!aa) {
                aa = true;
                continue;
            }
            String data[] = line.split("\\|");
            if (data.length >= 7) {
                SVNChanges change = new SVNChanges();
                change.revision = Integer.parseInt(data[0].trim());
                change.filePath = data[1].trim();
                change.changeType = data[2].trim();
                change.commitMessage = data[4].trim();
                change.author = data[5].trim();
                change.date = data[6].trim();

                changes.add(change);

            }
        }
        return changes;

        // Generated from
        // nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    public void saveBugFixRevisionList() throws FileNotFoundException, IOException {
        String path = cp.getDirectory() + "/commit_logs/commit_logs.txt";
        String bugRevisionFilePath = cp.getBugRevisionsFilePath();
        String bugFixRevisionFilePath = cp.getBugFixRevisionsFilePath();
        BufferedReader br = new BufferedReader(new FileReader(path));
        BufferedWriter bw1 = new BufferedWriter(new FileWriter(bugRevisionFilePath));
        BufferedWriter bw2 = new BufferedWriter(new FileWriter(bugFixRevisionFilePath));

        String line;
        boolean aa = false;
        String bugRevision = "";
        String bugFixRevision = "";

        while ((line = br.readLine()) != null) {
            if (!aa) {
                aa = true;
                continue;
            }
            String data[] = line.split("\\|");
            if (data.length >= 3) {
                String fixType = data[3].trim();
                if (fixType.equalsIgnoreCase("yes")) {
                    int currRevision = Integer.parseInt(data[0].trim());
                    int prevRevision = currRevision - 1;
                    bugRevision += prevRevision + " ";
                    bugFixRevision += currRevision + " ";
                }
            }
        }
        bw1.write(bugRevision + "\n");
        bw2.write(bugFixRevision + "\n");
        bw1.close();
        bw2.close();

    }

    public void print(Object o) {
        System.out.println(o);
    }

    public void saveBugFixRevisionInfo() throws FileNotFoundException, IOException {

        List<BugFix> totalBugFix = new ArrayList<>();
        List<Bug> totalBug = new ArrayList<>();

        for (int i = 2; i < cp.totalRevision; i++) {
            String path = cp.getChangeRevisionInfoPath(i);
            String bfPath = cp.getBugFixRevisionInfo(i);
            String bPath = cp.getBugRevisionInfo(i);
            List<BugRevisionInfo> bf = new ArrayList<>();

            BufferedReader reader = null;

            try {
                // Attempt to read the file for changes
                reader = new BufferedReader(new FileReader(path));

                String bfs = "";
                String bs = "";
                String line;
                int buggyRev = 0, bugFixRev = 0;
                boolean flag = false;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(" \\| ");
                    String fixType = parts[8].trim();

                    if (fixType.equalsIgnoreCase("yes")) {
                        flag = true;
                        buggyRev = Integer.parseInt(parts[1]);
                        String bgPath = parts[2].trim();
                        int[] prevLines = parseStartEndLine(parts[3]);
                        int bStartLine = prevLines[0];
                        int bEndLine = prevLines[1];

                        bugFixRev = Integer.parseInt(parts[5]);
                        String bgfPath = parts[6].trim();
                        int[] currentLines = parseStartEndLine(parts[7]);
                        int bfStartLine = currentLines[0];
                        int bfEndLine = currentLines[1];

                        bs += buggyRev + " | " + bgPath + " | " + bStartLine + " | " + bEndLine + "\n";
                        bfs += bugFixRev + " | " + bgfPath + " | " + bfStartLine + " | " + bfEndLine + "\n";

                    }
                }

                if (flag) {
                    Bug bug = new Bug();
                    bug.data = bs;
                    bug.revision = buggyRev;
                    totalBug.add(bug);

                    BugFix bugFix = new BugFix();
                    bugFix.data = bfs;
                    bugFix.revision = bugFixRev;
                    totalBugFix.add(bugFix);

                }
                // Write results to the respective files

            } catch (FileNotFoundException e) {
                // File doesn't exist, so skip and continue
                // System.err.println("File not found: " + path + ". Skipping this file.");
                continue;
            } catch (IOException e) {
                // Handle any I/O errors that occur during file processing
                e.printStackTrace();
            } finally {
                // Ensure readers and writers are closed properly
                try {
                    if (reader != null)
                        reader.close();

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        for (BugFix bugfix : totalBugFix) {
            BufferedWriter writer = null;
            try {
                String path = cp.getBugFixRevisionInfo(bugfix.revision);
                writer = new BufferedWriter(new FileWriter(path));
                writer.write(bugfix.data);
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (writer != null) {
                    writer.close();
                }
            }
        }

        for (Bug bug : totalBug) {
            BufferedWriter writer = null;
            try {
                String path = cp.getBugRevisionInfo(bug.revision);
                writer = new BufferedWriter(new FileWriter(path));
                writer.write(bug.data);
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (writer != null) {
                    writer.close();
                }
            }
        }

    }

    private int[] parseStartEndLine(String linePart) {
        String[] lines = linePart.split(",");
        int startLine = Integer.parseInt(lines[0]);
        int endLine = lines.length > 1 ? Integer.parseInt(lines[1]) : startLine; // If no end line, it's the same as
                                                                                 // start line
        return new int[] { startLine, endLine };
    }

    public List<BugFixRevisionInfo> getBugFixInfo(int revision) throws FileNotFoundException, IOException {
        List<BugFixRevisionInfo> bugfix = new ArrayList<>();
        String path = cp.getBugFixRevisionInfo(revision);
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        while ((line = br.readLine()) != null) {
            String data[] = line.split("\\|");
            if (data.length >= 4) {
                BugFixRevisionInfo bf = new BugFixRevisionInfo();
                bf.revision = Integer.parseInt(data[0]);
                bf.path = data[1].trim();
                bf.startLine = Integer.parseInt(data[2]);
                bf.endLine = Integer.parseInt(data[3]);
                bugfix.add(bf);
            }
        }
        return bugfix;
    }

    // get bug info

    public List<BugRevisionInfo> getBugInfo(int revision) throws FileNotFoundException, IOException {
        List<BugRevisionInfo> bug = new ArrayList<>();
        String path = cp.getBugRevisionInfo(revision);
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        while ((line = br.readLine()) != null) {
            String data[] = line.split("\\|");
            if (data.length >= 4) {
                BugRevisionInfo bf = new BugRevisionInfo();
                bf.revision = Integer.parseInt(data[0]);
                bf.path = data[1].trim();
                bf.startLine = Integer.parseInt(data[2]);
                bf.endLine = Integer.parseInt(data[3]);
                bug.add(bf);
            }
        }
        return bug;
    }

    public int getTotalBugFix(int start, int end) throws FileNotFoundException, IOException {
        int totalBugFix = 0;
        for (int i = 3; i < end; i++) {
            String path = cp.getBugFixRevisionInfo(i);
            BufferedReader br = new BufferedReader(new FileReader(path));
            String line;

            while ((line = br.readLine()) != null) {
                totalBugFix++;
            }
        }
        return totalBugFix;
    }

    public List<Integer> parseNumbersFromFile(String path) {
        List<Integer> numberList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            // Read file line by line
            while ((line = br.readLine()) != null) {
                // Split line into individual numbers based on spaces or other delimiters
                String[] numbers = line.split("\\s+");
                // Parse and add valid integers to the list
                for (String num : numbers) {
                    try {
                        numberList.add(Integer.parseInt(num));
                    } catch (NumberFormatException e) {
                        // Skip invalid (non-integer) values
                    }
                }
            }
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        }
        return numberList;
    }

    // Function to convert a list to a set (removes duplicates)
    public static Set<Integer> listToSet(List<Integer> numberList) {
        return new HashSet<>(numberList);
    }

    public int getTotalBug() throws FileNotFoundException, IOException {
        int totalBugNumber = 0;
        String filePath = cp.getBugRevisionsFilePath();
        List<Integer> numberList = parseNumbersFromFile(filePath);

        return numberList.size();
    }

    public Set<Integer> getBugList() {
        List<Integer> bugList = new ArrayList<>();
        String filePath = cp.getBugRevisionsFilePath();
        List<Integer> numberList = parseNumbersFromFile(filePath);
        Set<Integer> numberSet = listToSet(numberList);

        return numberSet;
    }

    public void generateOffsetFile(int currRevision) throws IOException {
        // Get changes between revisions
        List<CChanges> changes = getChanges(currRevision);

        if (changes == null || changes.isEmpty()) {
            print("I am NULL");
            return;
        }

        // File to store offsets
        String basePath = cp.getChangeRevisionOffsetPath(currRevision);
        File file = new File(basePath);

        // Ensure the file itself exists (creates empty file if not present)
        if (!file.exists()) {
            file.createNewFile();
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            writer.write(
                    "counter | prevPath | prevStartLine | prevEndLine | changeType | currPath | currStartLine | currEndLine | offsetStart | offsetEnd\n");

            int counter = 1;
            for (CChanges change : changes) {
                // Calculate line counts
                int prevLines = change.prevEndLine - change.prevStartLine + 1;
                int currLines = change.currEndLine - change.currStartLine + 1;

                // Compute offset
                int offset;
                switch (change.changeType.toLowerCase()) {
                    case "a":
                        offset = currLines;
                        break;
                    case "d":
                        offset = -prevLines;
                        break;
                    case "c":
                        offset = currLines - prevLines;
                        break;
                    default:
                        offset = 0;
                }

                // Offset start/end
                int offsetStart = change.currStartLine - change.prevStartLine;
                int offsetEnd = change.currEndLine - change.prevEndLine;

                // Write line
                String line = counter + " | " + change.prevPath + " | " + change.prevStartLine + " | "
                        + change.prevEndLine
                        + " | " + change.changeType + " | " + change.currPath + " | " + change.currStartLine + " | "
                        + change.currEndLine
                        + " | " + offsetStart + " | " + offsetEnd;
                writer.write(line + "\n");
                counter++;
            }
        }

        System.out.println("Offset file generated: " + file.getAbsolutePath());
    }

    public List<CChanges> getChanges(int revision) {
        List<CChanges> changesList = new ArrayList<>();
        String filePath = cp.getChangeRevisionInfoPath(revision);
        File file = new File(filePath);

        // Check if the file exists
        if (!file.exists()) {
            return null; // Return null if the file does not exist
        }

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty())
                    continue;

                // Split by pipe delimiter
                String[] parts = line.split("\\|");
                if (parts.length < 8)
                    continue; // Skip invalid lines

                CChanges change = new CChanges();

                // Parse basic fields
                change.serial = Integer.parseInt(parts[0].trim());
                change.prevRevision = Integer.parseInt(parts[1].trim());
                change.prevPath = parts[2].trim();

                // Parse prev start/end lines
                int[] prevLines = parseStartEndLine(parts[3].trim());
                change.prevStartLine = prevLines[0];
                change.prevEndLine = prevLines[1];

                change.changeType = parts[4].trim();
                change.currRevision = Integer.parseInt(parts[5].trim());
                change.currPath = parts[6].trim();

                // Parse curr start/end lines
                int[] currLines = parseStartEndLine(parts[7].trim());
                change.currStartLine = currLines[0];
                change.currEndLine = currLines[1];

                // Safely set fixType if exists
                if (parts.length > 8) {
                    change.fixType = parts[8].trim();
                } else {
                    change.fixType = ""; // default empty string
                }

                changesList.add(change);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (NumberFormatException e) {
            e.printStackTrace();
            System.out.println("Error parsing integer in change file: " + filePath);
        }

        return changesList;
    }

    public List<OffsetRevision> getOffsetByRevision(int revision) throws FileNotFoundException, IOException {
        List<OffsetRevision> records = new ArrayList<>();
        String filePath = cp.getChangeRevisionOffsetPath(revision);

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine(); // Skip header

            while ((line = br.readLine()) != null) {

                String[] parts = line.split("\\s*\\|\\s*");
                OffsetRevision o = new OffsetRevision();
                o.counter = Integer.parseInt(parts[0].trim());
                o.prevPath = parts[1].trim();
                o.prevStartLine = Integer.parseInt(parts[2].trim());
                o.prevEndLine = Integer.parseInt(parts[3].trim());
                o.changeType = parts[4].trim();
                o.currPath = parts[5].trim();
                o.currStartLine = Integer.parseInt(parts[6].trim());
                o.currEndLine = Integer.parseInt(parts[7].trim());
                o.offsetStart = Integer.parseInt(parts[8].trim());
                o.offsetEnd = Integer.parseInt(parts[9].trim());
                records.add(o);
            }
        }
        return records;
    }

    public Map<Integer, List<CChanges>> getAllChangesInRevisions(int startRevision, int endRevision) {
        Map<Integer, List<CChanges>> changesMap = new HashMap<>();
        System.out.println("End Revision : " + endRevision);
        for (int i = startRevision; i <= endRevision; i++) {
            List<CChanges> changes = getChanges(i);
            changesMap.put(i, changes);
        }

        return changesMap;
    }
}

class Change {
    int revision;
    String filePath;
    String fileName;
    String changeType;
    int startLine;
    int endLine;
}

class Loc {
    int revision;
    int lineCount;
}

class CloneMatcherOb {
    int index;
    int prevRevision;
    String prevFilePath;
    int prevStartLine;
    int prevEndLine; // Added for handling line ranges
    String changeType;
    int currRevision;
    String currFilePath;
    int currStartLine;
    int currEndLine; // Added for handling line ranges
}

class SVNChanges {
    int revision;
    String filePath;
    String fixType;
    String changeType;
    String commitMessage;
    String author;
    String date;

}

class BugRevisionInfo {
    int revision;
    String path;
    int startLine;
    int endLine;
}

class BugFixRevisionInfo {
    int revision;
    String path;
    int startLine;
    int endLine;
}

class BugFix {
    int revision;
    String data;
}

class Bug {
    int revision;
    String data;
}

class CChanges {
    int serial;
    int prevRevision;
    String prevPath;
    int prevStartLine;
    int prevEndLine;
    String changeType;
    int currRevision;
    String currPath;
    int currStartLine;
    int currEndLine;
    String fixType;
}

class Changes {
    int revision;
    String filePath;
    String fileName;
    String changeType;
    int startLine;
    int endLine;
    String fixType;

    public int getRevision() {
        return this.revision;
    }

}

class OffsetRevision {
    public int counter;
    public String prevPath;
    public int prevStartLine;
    public int prevEndLine;
    public String changeType;
    public String currPath;
    public int currStartLine;
    public int currEndLine;
    public int offsetStart;
    public int offsetEnd;

}

class Pair<K, V> {
    private K key;
    private V value;

    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public K getKey() {
        return key;
    }

    public V getValue() {
        return value;
    }

    // Override equals, hashCode, and toString methods as needed
    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        Pair<?, ?> pair = (Pair<?, ?>) o;
        if (!key.equals(pair.key))
            return false;
        return value.equals(pair.value);
    }

    @Override
    public int hashCode() {
        int result = key.hashCode();
        result = 31 * result + value.hashCode();
        return result;
    }

    @Override
    public String toString() {
        return "(" + key + ", " + value + ")";
    }
}