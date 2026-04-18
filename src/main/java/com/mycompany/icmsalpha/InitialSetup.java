/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 *
 * @author Administrator
 */
public class InitialSetup {

    CommonParameters cp = new CommonParameters();
    LoadParam lp = new LoadParam();
    // for methods
    String workFolder = "WorkFolder";

    String allMethodPath;
    String allMethodFolderName = "AllMethods";
    String globalMethodPath;
    String globalMethodFolderName = "GlobalMethods";
    String nonParsedMethodName = "NonParsedMethods";
    String nonParsedMethodPath;
    String offset = "Offset";
    String offsetPath;

    // for clones
    String cloneFolderPatht1b;
    String cloneFolderPatht2b;
    String cloneFolderPatht3b;
    String cloneFolderPatht1f;
    String cloneFolderPatht2f;
    String cloneFolderPatht3f;
    String allCloneFolderName = "AllClones";
    String globalCloneFolderName = "GlobalClones";
    String changeRevision = "ChangeRevisionInfo";
    String changeRevisionFolderPath;
    String revisionFolderPath;
    String dataset;
    String genealogy;

    public void initSetup() throws IOException {
        // WorkFolder is inside the icmsalpha project directory
        workFolder = System.getProperty("user.dir") + "/WorkFolder";

        dataset = workFolder + "/" + cp.getSubjectSystem();
        genealogy = workFolder + "/" + cp.getSubjectSystem() + "/Datasets";

        allMethodPath = workFolder + "/" + cp.getSubjectSystem() + "/Methods";
        globalMethodPath = workFolder + "/" + cp.getSubjectSystem() + "/Methods";
        offsetPath = workFolder + "/" + cp.getSubjectSystem() + "/Revisions/" + changeRevision;

        // clones
        cloneFolderPatht1b = workFolder + "/" + cp.getSubjectSystem() + "/Clones/Type1/Block";
        cloneFolderPatht2b = workFolder + "/" + cp.getSubjectSystem() + "/Clones/Type2/Block";
        cloneFolderPatht3b = workFolder + "/" + cp.getSubjectSystem() + "/Clones/Type3/Block";

        cloneFolderPatht1f = workFolder + "/" + cp.getSubjectSystem() + "/Clones/Type1/Function";
        cloneFolderPatht2f = workFolder + "/" + cp.getSubjectSystem() + "/Clones/Type2/Function";
        cloneFolderPatht3f = workFolder + "/" + cp.getSubjectSystem() + "/Clones/Type3/Function";
        revisionFolderPath = workFolder + "/" + cp.getSubjectSystem() + "/Revisions";
        changeRevisionFolderPath = workFolder + "/" + cp.getSubjectSystem() + "/Revisions/" + changeRevision;
        nonParsedMethodPath = workFolder + "/" + cp.getSubjectSystem() + "/Methods";

        Files.createDirectories(Paths.get(workFolder));
        createFolder(dataset, "Datasets");
        createFolder(genealogy, "CloneGenealogy");

        createFolder(allMethodPath, allMethodFolderName);
        createFolder(globalMethodPath, globalMethodFolderName);
        createFolder(nonParsedMethodPath, nonParsedMethodName);
        createFolder(offsetPath, offset);

        // for blocks

        // 1
        createFolder(cloneFolderPatht1b, allCloneFolderName);
        createFolder(cloneFolderPatht1b, globalCloneFolderName);
        createFolder(cloneFolderPatht1b, "GlobalCloneInfo");

        createFolder(cloneFolderPatht2b, allCloneFolderName);
        createFolder(cloneFolderPatht2b, globalCloneFolderName);
        createFolder(cloneFolderPatht2b, "GlobalCloneInfo");

        createFolder(cloneFolderPatht3b, allCloneFolderName);
        createFolder(cloneFolderPatht3b, globalCloneFolderName);
        createFolder(cloneFolderPatht3b, "GlobalCloneInfo");

        // for functions
        // type 1
        createFolder(cloneFolderPatht1f, allCloneFolderName);
        createFolder(cloneFolderPatht1f, globalCloneFolderName);
        createFolder(cloneFolderPatht1f, "GlobalCloneInfo");

        // type 2
        createFolder(cloneFolderPatht2f, allCloneFolderName);
        createFolder(cloneFolderPatht2f, globalCloneFolderName);
        createFolder(cloneFolderPatht2f, "GlobalCloneInfo");

        // type3
        createFolder(cloneFolderPatht3f, allCloneFolderName);
        createFolder(cloneFolderPatht3f, globalCloneFolderName);
        createFolder(cloneFolderPatht3f, "GlobalCloneInfo");

        createFolder(revisionFolderPath, changeRevision);
        // changeRevision Info
        // ChangeList , ChangeInfo
        createFolder(changeRevisionFolderPath, "ChangeList");
        createFolder(changeRevisionFolderPath, "ChangeInfo");

    }

    public static void createFolder(String path, String folderName) throws IOException {
        Path fullPath = Paths.get(path, folderName);

        if (!Files.exists(fullPath)) {
            Files.createDirectories(fullPath);
            System.out.println("Folder created: " + fullPath.toString());
        } else {
            System.out.println("Folder already exists: " + fullPath.toString());
        }
    }

}
