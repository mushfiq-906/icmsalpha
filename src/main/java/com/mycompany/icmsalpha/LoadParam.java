/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.Properties;

/**
 *
 * @author Administrator
 */
public class LoadParam {

    private static final String PARAM_FILE = "param.txt";
    private static final String CONFIG_FILE = "clone_config.properties";
    private String rootDir;
    private String startRevision;
    private String endRevision;
    private String lanExtension;
    private String cloneDir;

    // Clone detection thresholds from config
    private Properties cloneConfig = new Properties();

    LoadParam() {
        loadParams();
        loadCloneConfig();
    }

    private void loadParams() {
        try (BufferedReader reader = new BufferedReader(new FileReader(PARAM_FILE))) {
            rootDir = normalizeParamValue(reader.readLine());
            cloneDir = normalizeParamValue(reader.readLine());
            startRevision = reader.readLine();
            endRevision = reader.readLine();
            lanExtension = reader.readLine();
            warnIfMacPathOnWindows("Code Dir", rootDir);
            warnIfMacPathOnWindows("Clone Dir", cloneDir);

        } catch (IOException e) {
            // File doesn't exist or error reading, ignore and continue
        }
    }

    private String normalizeParamValue(String value) {
        if (value == null) {
            return null;
        }
        return value.trim();
    }

    private void warnIfMacPathOnWindows(String label, String path) {
        if (!PlatformSupport.isWindows() || path == null || path.isBlank()) {
            return;
        }

        if (path.startsWith("/Volumes/") || path.startsWith("/Users/") || path.startsWith("/opt/")) {
            System.out.println("Warning: " + label
                    + " in param.txt still points to a macOS path. Update it to a Windows path before running the analysis: "
                    + path);
        }
    }

    private void loadCloneConfig() {
        try (FileInputStream fis = new FileInputStream(CONFIG_FILE)) {
            cloneConfig.load(fis);
        } catch (IOException e) {
            // Config file not found, use defaults
            System.out.println("clone_config.properties not found, using default thresholds");
        }
    }

    public String getdir() {
        return rootDir;
    }

    public String getLanguage() {
        return lanExtension;
    }

    public int getLastRevNumber() {
        return Integer.parseInt(endRevision);
    }

    public int getFirstRevNumber() {
        return Integer.parseInt(startRevision);
    }

    public String getCloneDir() {
        return cloneDir;
    }

    public String getWorkFolder() {

        return "WorkFolder";

    }

    // ============ Clone Config Getters ============

    public double getType1Threshold() {
        return Double.parseDouble(cloneConfig.getProperty("type1.threshold", "0.99"));
    }

    public double getType2Threshold() {
        return Double.parseDouble(cloneConfig.getProperty("type2.threshold", "0.99"));
    }

    public double getType3Threshold() {
        return Double.parseDouble(cloneConfig.getProperty("type3.threshold", "0.70"));
    }

    public double getHighSimilarity() {
        return Double.parseDouble(cloneConfig.getProperty("high.similarity", "0.90"));
    }

    public int getLineTolerance() {
        return Integer.parseInt(cloneConfig.getProperty("line.tolerance", "5"));
    }

    public double getSpcpSimilarity() {
        return Double.parseDouble(cloneConfig.getProperty("spcp.similarity", "0.65"));
    }

    public double getSpcpMaxDrop() {
        return Double.parseDouble(cloneConfig.getProperty("spcp.max.drop", "0.20"));
    }

    public double getSpcpResync() {
        return Double.parseDouble(cloneConfig.getProperty("spcp.resync", "0.05"));
    }

    public int getSpcpMaxLagRevisions() {
        return Integer.parseInt(cloneConfig.getProperty("spcp.max.lag.revisions", "5"));
    }

}
