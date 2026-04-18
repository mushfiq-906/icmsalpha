/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

/**
 *
 * @author Administrator
 */
public class CommonParameters {
    LoadParam lp = new LoadParam();
    // AccessPoint ap = new AccessPoint();
    String subjectSystem = lp.getdir();
    int totalRevision = lp.getLastRevNumber();
    int firstRevision = lp.getFirstRevNumber();
    public String WorkFolder = "";

    public CommonParameters() {
        // WorkFolder is inside the icmsalpha project directory
        WorkFolder = System.getProperty("user.dir") + "/WorkFolder/";
    }

    // int lowestRevision = ap.firstGenRevision();
    public String getDirectory() {
        return lp.getdir();
    }

    public String getSubjectSystem() {

        int lastIndex = lp.getdir().lastIndexOf('/');
        if (lastIndex == -1)
            lastIndex = lp.getdir().lastIndexOf('\\'); // Fallback
        // Extract the substring after the last slash
        this.subjectSystem = lp.getdir().substring(lastIndex + 1);
        return subjectSystem;
    }

    public String getCloneDir() {
        return lp.getCloneDir();
    }

    public String getLan() {
        return lp.getLanguage();
    }

    public String getRevisionPath(int revision) {
        String revisionPath = this.getDirectory() + "/Revision_" + revision;
        return revisionPath;
    }

    public String getT1BClones(int revision) {

        return WorkFolder + getSubjectSystem() + "/Clones/Type1/Block/AllClones/" + revision + "_clones.txt";
    }

    public String getT1FClones(int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/Type1/Function/AllClones/" + revision + "_clones.txt";
    }

    public String getT2BClones(int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/Type2/Block/AllClones/" + revision + "_clones.txt";
    }

    public String getT2FClones(int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/Type2/Function/AllClones/" + revision + "_clones.txt";
    }

    public String getT3BClones(int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/Type3/Block/AllClones/" + revision + "_clones.txt";
    }

    public String getT3FClones(int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/Type3/Function/AllClones/" + revision + "_clones.txt";
    }

    // global clone
    public String getT1BGlobalClones(int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/Type1/Block/GlobalClones/" + revision + "_clones.txt";
    }

    public String getT1FGlobalClones(int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/Type1/Function/GlobalClones/" + revision + "_clones.txt";
    }

    public String getT2BGlobalClones(int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/Type2/Block/GlobalClones/" + revision + "_clones.txt";
    }

    public String getT2FGlobalClones(int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/Type2/Function/GlobalClones/" + revision + "_clones.txt";
    }

    public String getDatabaseAdmin() {
        return "root";

    }

    public String getDatabasePassword() {
        return "";
    }

    // public String getBugRevisionsFilePath(){
    // return WorkFolder + getSubjectSystem() +
    // "\\Revisions\\AllBugRevision\\BugRevision.txt";
    // }
    //
    // public String getBugFixRevisionsFilePath(){
    // return WorkFolder + getSubjectSystem() +
    // "\\Revisions\\AllBugRevision\\BugFixRevision.txt";
    // }

    public String getBugFixRevisionsFilePath() {
        return WorkFolder + getSubjectSystem() + "/Revisions/BugFixRevisionInfo/BugFixList/BugFixRevision.txt";
    }

    public String getBugRevisionsFilePath() {
        return WorkFolder + getSubjectSystem() + "/Revisions/BugRevisionInfo/BugList/BugRevision.txt";
    }

    public String getLogFile() {
        return "ChangeLog.txt";
    }

    public String getBugRevisionInfo() {
        return WorkFolder + getSubjectSystem() + "/Revisions/AllBugRevision/BugRevisionInfo.txt";
    }

    public String getAllMethodsPath(int revision) {
        return WorkFolder + getSubjectSystem() + "/Methods/AllMethods/" + revision + "_methods.txt";
    }

    public String getGlobalMethodsPath(int revision) {
        return WorkFolder + getSubjectSystem() + "/Methods/GlobalMethods/" + revision + "_methods.txt";
    }

    public String NonParsedMethodsPath(int revision) {
        return WorkFolder + getSubjectSystem() + "/Methods/NonParsedMethods/" + revision + "_methods.txt";
    }

    public String getBugInfo() {
        return WorkFolder + getSubjectSystem() + "/Bugs/AllBugInfo/BugCodeInfo.txt";
    }

    public String getRevisionToMaxGlobal(String cChoice, String granularity) {
        return WorkFolder + getSubjectSystem() + "/Clones/" + cChoice + "/" + granularity
                + "/Evaluation+/RevisionToMaxGlobal.txt";
    }

    public String getEvaluation(String cChoice, String granularity) {
        return WorkFolder + getSubjectSystem() + "/Clones/" + cChoice + "/" + granularity + "/Evaluation";
    }

    public String getGenealogyPath(String choice, String granularuty) {
        return WorkFolder + getSubjectSystem() + "/Datasets/CloneGenealogy/" + choice + "_" + granularuty + ".csv";
    }

    public String getBugFixInfo() {
        return WorkFolder + getSubjectSystem() + "/Bugs/AllBugInfo/BugFixCodeInfo.txt";
    }

    String getBugFixChangesPath() {
        return WorkFolder + getSubjectSystem() + "/Bugs/BugFixChangeBefore";
    }

    String getChangesFilePath(int revision) {
        return null;
    }

    String getChangesPath(int revision) {
        return WorkFolder + getSubjectSystem() + "/Bugs/BugFixChangeBefore/" + revision + "_changes.txt";
    }

    String getGlobalClonePath(String cChoice, String granularity, int revision) {

        return WorkFolder + getSubjectSystem() + "/Clones/" + cChoice + "/" + granularity + "/GlobalClones/" + revision
                + "_clones.txt";
    }

    /**
     * Get path to GlobalCloneInfo output file (clone-method linking)
     */
    public String getGlobalCloneInfoPath(String cloneType, String granularity, int revision) {
        return WorkFolder + getSubjectSystem() + "/Clones/" + cloneType + "/" + granularity + "/GlobalCloneInfo/"
                + revision + "_clones.txt";
    }

    String getBugRevisionChangesFile() {
        return WorkFolder + getSubjectSystem() + "/Revisions/AllBugRevision/BugRevision.txt";// Generated from
                                                                                             // nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    String getBugFixRevisionInfo() {
        return WorkFolder + getSubjectSystem() + "/Bugs/AllBugInfo/BugFixCodeInfo.txt";
    }

    String getBugAnalyzerInitialPath(int revision) {
        return WorkFolder + getSubjectSystem() + "/Bugs/Analyzer/Initial/" + revision + "_bugs.txt"; // Generated from
                                                                                                     // nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    String getBugAnalyzerSeverityPath(int revision) {
        return WorkFolder + getSubjectSystem() + "/Bugs/Analyzer/BugSeverity/" + revision + "_bugs.txt"; // Generated
                                                                                                         // from
                                                                                                         // nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody

    }

    String getRevisionLoc() {
        return "WorkFolder/jEdit/Revisions/loc_info.txt"; // Generated from
                                                          // nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    String getCloneLOCPath(String cloneType, String granularity) {
        return WorkFolder + getSubjectSystem() + "/Clones/" + cloneType + "/" + granularity + "/micro_loc.txt";
    }

    String getCCNMethodsRevision(int revision) {
        return WorkFolder + getSubjectSystem() + "/Methods/CCNMethods/" + revision + "_methods.txt";
    }

    String getAllChangeRevisionInfoPath() {
        return WorkFolder + getSubjectSystem() + "/Revisions/AllChangeRevisionInfo/ChangeRevision.txt";
    }

    String getChangeRevisionOffsetPath(int revision) {// ChangeRevisionInfo/Offset
        return WorkFolder + getSubjectSystem() + "/Revisions/ChangeRevisionInfo/Offset/revision_" + revision
                + ".txt";

        // Generated from
        // nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    String getChangeRevisionInfoPath(int revision) {
        return WorkFolder + getSubjectSystem() + "/Revisions/ChangeRevisionInfo/ChangeInfo/revision_" + revision
                + ".txt";

        // Generated from
        // nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    String getBugFixRevisionInfo(int revision) {
        return WorkFolder + getSubjectSystem() + "/Revisions/BugFixRevisionInfo/BugFixInfo/revision_" + revision
                + ".txt";
    }

    String getBugRevisionInfo(int revision) {
        return WorkFolder + getSubjectSystem() + "/Revisions/BugRevisionInfo/BugInfo/revision_" + revision + ".txt";
    }

}
