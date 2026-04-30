
package com.mycompany.icmsalpha;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Parzival
 */
public class Utility {

    public void RunBash(String url, String download_path) {
        Path scriptPath = Paths.get("Scripts", "git_script.sh").toAbsolutePath().normalize();
        if (!Files.exists(scriptPath)) {
            System.err.println("Script not found: " + scriptPath);
            return;
        }

        try {
            ProcessBuilder processBuilder = new ProcessBuilder(buildBashCommand(scriptPath, url, download_path));
            Process process = processBuilder.start();

            // Wait for the process to finish
            int exitCode = process.waitFor();

            if (exitCode == 0) {
                System.out.println("Script executed successfully!");
            } else {
                System.err.println("Failed to execute script. Exit code: " + exitCode);
            }

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    private List<String> buildBashCommand(Path scriptPath, String url, String downloadPath) {
        List<String> command = new ArrayList<>();

        if (PlatformSupport.isWindows()) {
            Path gitBash = Paths.get("C:\\Program Files\\Git\\bin\\bash.exe");
            command.add(Files.exists(gitBash) ? gitBash.toString() : "bash");
        } else {
            command.add("bash");
        }

        command.add(scriptPath.toString());
        command.add(url);
        command.add(downloadPath);
        return command;
    }
     
    public void Print(){
        System.out.print("I am a method ");
    }
    
    public static boolean isThisTwoCodeSimilar(String code1, String code2 , double threshold) {
           double similarity1 = calculateSimilarity(code1, code2);
            double similarity2 = calculateSimilarity(code2, code1);
            double maxSimilarity = Math.min(similarity1, similarity2);
            return maxSimilarity >= threshold/100;
    }
    
    public static double calculateSimilarity(String code1, String code2) {
        Map<String, Integer> wordFrequency1 = getWordFrequency(code1);
        Map<String, Integer> wordFrequency2 = getWordFrequency(code2);

        double dotProduct = calculateDotProduct(wordFrequency1, wordFrequency2);
        double magnitude1 = calculateMagnitude(wordFrequency1);
        double magnitude2 = calculateMagnitude(wordFrequency2);

        if (magnitude1 == 0 || magnitude2 == 0) {
            return 0.0; // Avoid division by zero
        }

        return dotProduct / (magnitude1 * magnitude2);
    }

    private static Map<String, Integer> getWordFrequency(String code) {
        Map<String, Integer> wordFrequency = new HashMap<>();
        String[] words = code.split("\\W+"); // Split by non-word characters

        for (String word : words) {
            wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
        }

        return wordFrequency;
    }

    private static double calculateDotProduct(Map<String, Integer> wordFrequency1, Map<String, Integer> wordFrequency2) {
        double dotProduct = 0.0;

        for (Map.Entry<String, Integer> entry : wordFrequency1.entrySet()) {
            String word = entry.getKey();
            int frequency1 = entry.getValue();
            int frequency2 = wordFrequency2.getOrDefault(word, 0);
            dotProduct += frequency1 * frequency2;
        }

        return dotProduct;
    }

    private static double calculateMagnitude(Map<String, Integer> wordFrequency) {
        double magnitude = 0.0;

        for (int frequency : wordFrequency.values()) {
            magnitude += Math.pow(frequency, 2);
        }

        return Math.sqrt(magnitude);
    }  
}
