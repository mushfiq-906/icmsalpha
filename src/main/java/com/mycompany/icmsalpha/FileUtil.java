/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author Administrator
 */
public class FileUtil {
     /**
     * Reads the content of a file from startLine to endLine (inclusive) and returns it as a single String.
     *
     * @param filePath  Path to the file
     * @param startLine Starting line number (1-based)
     * @param endLine   Ending line number (inclusive, 1-based)
     * @return Content from startLine to endLine as a String
     * @throws IOException if file not found or read fails
     */
    public static String getFileContentByLineRange(String filePath, int startLine, int endLine) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int currentLine = 0;
            while ((line = br.readLine()) != null) {
                currentLine++;
                if (currentLine >= startLine && currentLine <= endLine) {
                    sb.append(line).append(System.lineSeparator());
                }
                if (currentLine > endLine) break;
            }
        }
        return sb.toString();
    }

    
}
