/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.icmsalpha;

/**
 *
 * @author Parzival
 */
import java.io.File;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class XMLParser {
    public  void parseMe(String path) {
        try {
            // Load the XML file
            File xmlFile = new File(path);
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(xmlFile);
            doc.getDocumentElement().normalize();
            
            // Get all class elements
            NodeList classList = doc.getElementsByTagName("class");
            for (int i = 0; i < classList.getLength(); i++) {
                Element classElement = (Element) classList.item(i);
                
                // Extract class attributes
                String classId = classElement.getAttribute("classid");
                int nClones = Integer.parseInt(classElement.getAttribute("nclones"));
                int nLines = Integer.parseInt(classElement.getAttribute("nlines"));
                int similarity = Integer.parseInt(classElement.getAttribute("similarity"));
                
                System.out.println("Class ID: " + classId);
                System.out.println("Number of Clones: " + nClones);
                System.out.println("Number of Lines: " + nLines);
                System.out.println("Similarity: " + similarity);
                
                // Get all source elements within the class
                NodeList sourceList = classElement.getElementsByTagName("source");
                for (int j = 0; j < sourceList.getLength(); j++) {
                    Element sourceElement = (Element) sourceList.item(j);
                    
                    // Extract source attributes
                    String file = sourceElement.getAttribute("file");
                    int startLine = Integer.parseInt(sourceElement.getAttribute("startline"));
                    int endLine = Integer.parseInt(sourceElement.getAttribute("endline"));
                    int pcid = Integer.parseInt(sourceElement.getAttribute("pcid"));
                    
                    System.out.println("Source File: " + file);
                    System.out.println("Start Line: " + startLine);
                    System.out.println("End Line: " + endLine);
                    System.out.println("PCID: " + pcid);
                }
                System.out.println(); // Add a separator between classes
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
