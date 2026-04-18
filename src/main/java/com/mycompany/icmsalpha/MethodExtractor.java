/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

/**
 *
 * @author Administrator
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MethodExtractor {
    static int methodId = 1;
    private int revision;
    private String revisionPath;
    private String extension;
    private String language;
    private String tagsPath;
    private String tagsOutputFile;
    private String outputPath;
    CommonParameters cp = new CommonParameters();
    LoadParam lp = new LoadParam();
    private int totalRevision = cp.totalRevision;
    private int firstRevision = cp.firstRevision;

    public void start() throws IOException {
        for (int i = firstRevision; i <= totalRevision; i++) {
            System.out.println("Working Revision : " + i);
            extractMethod(i);
        }
        System.out.println("Method Parsing Successfull \n\n Now You can Press Global Analysis Button !!");
    }

    public void extractMethod(int revision) throws IOException {

        this.revision = revision;
        this.revisionPath = cp.getRevisionPath(revision);
        this.extension = cp.getLan();
        // System.out.println(this.extension);
        this.tagsOutputFile = cp.NonParsedMethodsPath(revision); // Specify your output file for tags
        this.outputPath = cp.getAllMethodsPath(revision);
        if (extension.equals("java")) {
            this.language = "Java";
        }

        if (extension.equals("c")) {
            this.language = "C";
        }

        this.tagsPath = "tags";

        // System.out.println(this.revision + " " + this.revisionPath + " "
        // +this.tagsOutputFile + " " + this.outputPath);
        // System.out.println("PROCESSING NON PARSING USING CTAGS");
        runProcess();
        // System.out.println("CTAGS Parsing completed");
        // System.out.println("Initiate method parsing");

        runTagFileParser1(this.extension);
    }

    public void runProcess() {
        String ctagsPath = "/opt/homebrew/bin/ctags"; // Universal Ctags (Homebrew) - must use absolute path

        // Ensure output directory exists before running ctags
        java.io.File tagsFile = new java.io.File(tagsOutputFile);
        tagsFile.getParentFile().mkdirs();

        // Also ensure AllMethods directory exists
        java.io.File outputFile = new java.io.File(outputPath);
        outputFile.getParentFile().mkdirs();

        // using universal ctags --fields=+ne -R --languages=
        String command = ctagsPath + " --quiet --fields=+n -R --languages=" + language + " -o " + tagsOutputFile + " "
                + revisionPath;

        try {
            // Create ProcessBuilder
            ProcessBuilder pb = new ProcessBuilder(command.split("\\s+"));
            pb.inheritIO(); // Redirects the subprocess's standard input, output, and error streams to the
                            // Java process

            // Start the process
            Process process = pb.start();

            // Wait for the process to finish
            int exitCode = process.waitFor();

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public void runTagFileParser(String lan) throws IOException {

        BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath));
        BufferedReader br = new BufferedReader(new FileReader(tagsOutputFile));

        String line;
        String regexPattern = null;
        if (lan.equals("java")) {
            System.out.println(" JAVAAAAAAAAAAAAAAAAAAAA");
            regexPattern = "^(\\w+)\\s+([^\\s]+)\\s+/\\^([\\s\\S]*?)\\$/;\"\\t(m)\\tline:(\\d+)(?:\\t(class|interface|enum):(\\w+))?.*$";
        } else if (lan.equals("c")) {

            System.out.println("CCCCCCCCCCCCCCCCCCCCCC");
            regexPattern = "^(\\w+)\\s+([^\\s]+)\\s+/\\^([\\s\\S]*?)\\$/;\"\\t(m)\\tline:(\\d+)(?:\\t(class|interface|enum):(\\w+))?.*$";

        }
        Pattern TAG_LINE_PATTERN = Pattern.compile(regexPattern);
        Matcher matcher;

        boolean firstLine = true; // To skip writing header after first line

        try {
            while ((line = br.readLine()) != null) {
                matcher = TAG_LINE_PATTERN.matcher(line);

                if (matcher.find()) {
                    String name = matcher.group(1);
                    String filePath = matcher.group(2);
                    String signature = matcher.group(3);
                    String type = matcher.group(4);
                    int startLine = Integer.parseInt(matcher.group(5));
                    if (!type.equals("m")) {
                        continue;
                    }
                    int endLine = retrieveEndLine(filePath, startLine, signature);
                    // Construct the string to write to file
                    String ss = revision + " | " + name + " | " + signature + " | " + filePath + " | " + startLine
                            + " | " + endLine + " | " + methodId + "\n";
                    methodId++;
                    if (!firstLine) {
                        bw.write(ss);
                    } else {
                        firstLine = false;
                        // Uncomment to write header only once
                        bw.write(
                                "revision_number | method_name | method_signature | src | start_line | end_line | methodId\n");
                        bw.write(ss); // Write the first line of data
                    }
                }
            }
        } finally {
            // Close resources in finally block
            if (bw != null) {
                bw.close();
            }
            if (br != null) {
                br.close();
            }
        }

        System.out.println("Parsing completed. Check tagresult.txt for output.");
    }

    public void runTagFileParser1(String lan) throws IOException {

        BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath));
        BufferedReader br = new BufferedReader(new FileReader(tagsOutputFile));

        String line;
        String regexPattern;

        // -------- REGEX FOR JAVA LANGUAGE --------
        // Ctags output format with TABS:
        // name<TAB>filepath<TAB>/^signature$/;"<TAB>type<TAB>line:N[<TAB>class:classname]
        // Use \\t for explicit tab matching
        String JAVA_REGEX = "^(\\w+)\\t(\\S+)\\t/\\^([\\s\\S]*?)\\$/;\"\\t(\\w)\\t+line:(\\d+)"
                + "(?:\\t(?:class|interface|enum):(\\w+))?.*$";

        // -------- REGEX FOR C LANGUAGE --------
        // C uses same tab-separated format, but some ctags versions output simpler
        // format
        // Primary: name<TAB>filepath<TAB>/^signature$/;"<TAB>type<TAB>line:N
        String C_REGEX = "^(\\w+)\\t(\\S+)\\t/\\^([\\s\\S]*?)\\$/;\"\\t(\\w)\\t+line:(\\d+)";

        // Alternative C format: name<TAB>filepath<TAB>line#;"<TAB>type (for some ctags
        // versions)
        String C_REGEX_ALT = "^(\\w+)\\t(\\S+)\\t(\\d+);\"\\t(\\w)";

        // pick pattern based on language
        regexPattern = lan.equals("c") ? C_REGEX : JAVA_REGEX;

        Pattern TAG_LINE_PATTERN = Pattern.compile(regexPattern);
        Pattern TAG_LINE_PATTERN_ALT = lan.equals("c") ? Pattern.compile(C_REGEX_ALT) : null;
        Matcher matcher;

        boolean firstLine = true;
        int matchCount = 0;
        int methodCount = 0;
        int skippedLines = 0;

        try {
            while ((line = br.readLine()) != null) {

                // Skip metadata lines starting with !
                if (line.startsWith("!"))
                    continue;

                // Skip lines that are clearly not function/method entries
                if (!line.contains("\t")) {
                    continue;
                }

                matcher = TAG_LINE_PATTERN.matcher(line);
                boolean matched = matcher.find();

                // Try alternative pattern for C if primary doesn't match
                boolean useAltPattern = false;
                if (!matched && TAG_LINE_PATTERN_ALT != null) {
                    matcher = TAG_LINE_PATTERN_ALT.matcher(line);
                    matched = matcher.find();
                    useAltPattern = matched;
                }

                if (!matched) {
                    skippedLines++;
                    continue;
                }

                matchCount++;

                // Extract groups based on which pattern matched
                String name = matcher.group(1);
                String filePath = matcher.group(2);
                String signature;
                String type;
                int startLine;
                String className = "";

                if (useAltPattern) {
                    // Alternative pattern: name<TAB>filepath<TAB>lineNum;"<TAB>type
                    // Groups: (name) (filepath) (lineNum) (type)
                    signature = name + "()"; // Placeholder signature for alt pattern
                    startLine = Integer.parseInt(matcher.group(3));
                    type = matcher.group(4);
                } else {
                    // Primary pattern: name<TAB>filepath<TAB>/^signature$/;"<TAB>type<TAB>line:N
                    // Groups: (name) (filepath) (signature) (type) (lineNum) [className]
                    signature = matcher.group(3);
                    type = matcher.group(4);
                    startLine = Integer.parseInt(matcher.group(5));
                    // For Java, extract class name from group 6 if available
                    if (lan.equals("java") && matcher.groupCount() >= 6 && matcher.group(6) != null) {
                        className = matcher.group(6);
                    }
                }

                // -------- Filter: Accept methods (type 'm' for Java) or functions (type 'f'
                // for C) --------
                // 'f' = C function, 'm' = Java method, 'c' = class
                boolean isMethod = type.equals("m"); // Java method
                boolean isCFunction = type.equals("f") && lan.equals("c"); // C function
                if (!isMethod && !isCFunction) {
                    continue;
                }

                // -------- Filter 2: signature must contain parentheses --------
                // Skip for alt pattern since we added placeholder
                if (!useAltPattern && (!signature.contains("(") || !signature.contains(")"))) {
                    continue;
                }

                // -------- For Java, skip if signature looks like a variable declaration
                // --------
                if (lan.equals("java") && !useAltPattern) {
                    // Skip if contains '=' (variable initialization)
                    if (signature.contains("="))
                        continue;
                    // Skip if it's an array declaration
                    if (signature.contains("[") && signature.contains("]"))
                        continue;
                }

                methodCount++;

                // Compute ending line
                int endLine = retrieveEndLine(filePath, startLine, signature);

                // -------- Clean up signature --------
                // Remove trailing '{' and whitespace from signature
                String cleanSignature = signature.trim();
                if (cleanSignature.endsWith("{")) {
                    cleanSignature = cleanSignature.substring(0, cleanSignature.length() - 1).trim();
                }

                // -------- Extract package name (for Java) --------
                String packageName = "";
                if (lan.equals("java")) {
                    packageName = extractPackageName(filePath);
                }

                // -------- Extract relative source path --------
                // Convert full path like
                // /Volumes/Thesis/Nicad-7.0.1/systems/dnsjava/Revision_2/DNS/File.java
                // to relative path like DNS/File.java (after Revision_X/)
                String relativeSrc = extractRelativePath(filePath);

                // -------- For C, extract class name from file name (no package) --------
                if (lan.equals("c") && className.isEmpty()) {
                    // Use file name without extension as a pseudo-class name
                    String fileName = new java.io.File(filePath).getName();
                    if (fileName.contains(".")) {
                        className = fileName.substring(0, fileName.lastIndexOf('.'));
                    } else {
                        className = fileName;
                    }
                }

                // -------- For Java, if class name is empty, try to get from file name --------
                if (lan.equals("java") && className.isEmpty()) {
                    String fileName = new java.io.File(filePath).getName();
                    if (fileName.endsWith(".java")) {
                        className = fileName.substring(0, fileName.length() - 5);
                    }
                }

                String output = revision + " | " + name + " | " + cleanSignature + " | " +
                        packageName + " | " + className + " | " + relativeSrc + " | " + startLine + " | " +
                        endLine + " | " + methodId + "\n";

                methodId++;

                // Write header once
                if (firstLine) {
                    bw.write(
                            "revision_number | method_name | method_signature | package_name | class_name | src | start_line | end_line | methodId\n");
                    firstLine = false;
                }

                bw.write(output);
            }

        } finally {
            bw.close();
            br.close();
        }

        System.out.println("Parsing completed for revision " + revision + ". Found " + methodCount + " methods from "
                + matchCount + " regex matches. Skipped " + skippedLines + " non-matching lines.");
    }

    /**
     * Extract package name from a Java source file
     */
    private String extractPackageName(String filePath) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                // Skip empty lines and comments
                if (line.isEmpty() || line.startsWith("//") || line.startsWith("/*") || line.startsWith("*")) {
                    continue;
                }
                // Look for package declaration
                if (line.startsWith("package ")) {
                    // Extract package name: "package com.example.foo;" -> "com.example.foo"
                    String pkg = line.substring(8).trim();
                    if (pkg.endsWith(";")) {
                        pkg = pkg.substring(0, pkg.length() - 1).trim();
                    }
                    return pkg;
                }
                // If we hit an import or class declaration, there's no package
                if (line.startsWith("import ") || line.contains("class ") || line.contains("interface ")
                        || line.contains("enum ")) {
                    break;
                }
            }
        } catch (IOException e) {
            // Silently return empty string if file cannot be read
        }
        return "";
    }

    /**
     * Extract relative path after Revision_X/ prefix
     * e.g., /Volumes/Thesis/.../Revision_2/DNS/File.java -> DNS/File.java
     */
    private String extractRelativePath(String fullPath) {
        // Normalize path separators
        String normalizedPath = fullPath.replace("\\", "/");

        // Look for Revision_N/ pattern and extract everything after it
        Pattern revPattern = Pattern.compile("Revision_\\d+/(.+)$");
        Matcher m = revPattern.matcher(normalizedPath);
        if (m.find()) {
            return m.group(1);
        }

        // Fallback: just return the file name if no Revision_ pattern found
        return new java.io.File(fullPath).getName();
    }

    private boolean isValidFunctionSignature(String sig) {

        String s = sig.trim();

        // Must contain parentheses
        if (!s.contains("(") || !s.contains(")"))
            return false;

        // Exclude arrays, bitfields, assignments, initializers
        if (s.contains("[") || s.contains("]"))
            return false; // global arrays
        if (s.contains("="))
            return false; // declarations/initializers
        if (s.contains(":"))
            return false; // bitfields
        if (s.startsWith("{") || s.startsWith("}"))
            return false;

        // Exclude struct/enum/typedef declarations
        if (s.startsWith("struct ") ||
                s.startsWith("enum ") ||
                s.startsWith("typedef"))
            return false;

        // Only accept something that looks like: return-type name(args)
        // e.g., "int add(int a, int b)"
        return s.matches(".*\\w+\\s*\\([^)]*\\)");
    }

    public void print(Object o) {
        System.out.println(o);
    }

    private static int retrieveEndLine(String filePath, int startLine, String methodSignature) throws IOException {
        // Read all lines from the file
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        int currentLine = 0;
        int openBraces = 0;
        int endLine = startLine;

        while ((line = reader.readLine()) != null) {
            currentLine++;

            // Skip lines before the start line
            if (currentLine < startLine) {
                continue;
            }

            // Count braces to find the end of the method/class
            openBraces += countOccurrences(line, '{');
            openBraces -= countOccurrences(line, '}');

            if (openBraces == 0 && currentLine > startLine) {
                endLine = currentLine;
                break;
            }
        }

        reader.close();
        return endLine;
    }

    private static int countOccurrences(String line, char ch) {
        int count = 0;
        for (char c : line.toCharArray()) {
            if (c == ch) {
                count++;
            }
        }
        return count;
    }

}
