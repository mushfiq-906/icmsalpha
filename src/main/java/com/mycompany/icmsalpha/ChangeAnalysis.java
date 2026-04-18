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
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class ChangeAnalysis {
    AccessPoint ap = new AccessPoint();
    CommonParameters cp = new CommonParameters();
    LoadParam lp = new LoadParam();

    public ChangeAnalysis() {
    }

    public void print(Object o) {
        System.out.println(o);
    }

    public void giveMeOffset() throws IOException {
        List<SVNChanges> changeByRevision = ap.getSVNChanges();

        if (changeByRevision == null || changeByRevision.isEmpty()) {
            print("No SVN changes found.");
            return;
        }

        TreeMap<Integer, List<SVNChanges>> sortedChanges = changeByRevision.stream()
                .collect(Collectors.groupingBy(change -> change.revision, TreeMap::new, Collectors.toList()));

        print("I am inside init");

        boolean firstRevisionFlag = true;

        boolean flag = false;
        for (Map.Entry<Integer, List<SVNChanges>> entry : sortedChanges.entrySet()) {
            int revision = entry.getKey();

            if (!flag) {
                flag = true;
                continue;
            }
            ap.generateOffsetFile(revision);

        }

        print("getMapping");
    }

    public void init() throws IOException {
        List<SVNChanges> changeByRevision = ap.getSVNChanges();

        if (changeByRevision == null || changeByRevision.isEmpty()) {
            print("No SVN changes found.");
            return;
        }

        TreeMap<Integer, List<SVNChanges>> sortedChanges = changeByRevision.stream()
                .collect(Collectors.groupingBy(change -> change.revision, TreeMap::new, Collectors.toList()));

        print("I am inside init");

        boolean firstRevisionFlag = true;

        for (Map.Entry<Integer, List<SVNChanges>> entry : sortedChanges.entrySet()) {
            int revision = entry.getKey();

            if (revision >= lp.getLastRevNumber())
                break;

            String fixType = entry.getValue().stream()
                    .map(c -> c.fixType)
                    .filter(Objects::nonNull)
                    .findFirst()
                    .orElse(" ");

            List<String> revisionFiles = entry.getValue().stream()
                    .map(c -> parseFilePath(c.filePath))
                    .collect(Collectors.toList());

            print("Working Revision :::: " + revision);

            start(revisionFiles, revision, fixType, firstRevisionFlag);

            // After the first revision is processed, set flag to false
            firstRevisionFlag = false;
        }

        print("getMapping");
    }

    void start(List<String> paths, int currRevision, String fixType, boolean isFirstRevision) throws IOException {
        int counter = 1;
        String revisionFilePath = cp.getChangeRevisionInfoPath(currRevision);
        Set<String> uniqueChanges = new HashSet<>();

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(revisionFilePath))) {

            // Handle first revision using flag
            if (isFirstRevision) {
                for (String filePath : paths) {
                    String changeStr = String.format("0 | %s | 0 | a | %d | %s | 1-EOF",
                            "N/A", currRevision, getRevPath(currRevision, filePath));
                    if (uniqueChanges.add(changeStr)) {
                        print(changeStr);
                        bw.write(counter + " | " + changeStr + " | " + fixType + "\n");
                        counter++;
                    }
                }

                return; // No need to run diff for first revision
            }

            // Normal case: diff with previous revision
            for (String filePath : paths) {
                int prevRevision = currRevision - 1;
                String currPath = getFullPath(currRevision, filePath);
                String prevPath = getFullPath(prevRevision, filePath);

                try {
                    ProcessBuilder pb = new ProcessBuilder(
                            "diff",
                            prevPath, currPath);
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            line = line.trim();
                            if (line.startsWith("<") || line.startsWith(">"))
                                continue;

                            if (line.matches("\\d+(,\\d+)?[acd]\\d+(,\\d+)?")) {
                                String[] parts = line.split("[acd]");
                                String prevLine = parts[0].trim();
                                String currLine = parts[1].trim();

                                char type = line.contains("c") ? 'c' : line.contains("a") ? 'a' : 'd';
                                String changeStr = String.format("%d | %s | %s | %c | %d | %s | %s",
                                        prevRevision, getRevPath(prevRevision, filePath), prevLine,
                                        type, currRevision, getRevPath(currRevision, filePath), currLine);

                                if (uniqueChanges.add(changeStr)) {
                                    print(changeStr);
                                    bw.write(counter + " | " + changeStr + " | " + fixType + "\n");
                                    counter++;
                                }
                            }
                        }
                    }
                    process.waitFor();

                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
            }

        }

    }

    public String getRevPath(int revision, String filepath) {
        return "Revision_" + revision + "/" + filepath;
    }

    public String getFullPath(int revision, String filepath) {
        return cp.getCloneDir() + "/Revision_" + revision + "/" + filepath;
    }

    private String parseFilePath(String fullPath) {
        int revisionIndex = fullPath.indexOf("Revision_");
        if (revisionIndex != -1) {
            int pathStartIndex = fullPath.indexOf('/', revisionIndex);
            if (pathStartIndex != -1)
                return fullPath.substring(pathStartIndex + 1);
        }
        return fullPath;
    }

}
