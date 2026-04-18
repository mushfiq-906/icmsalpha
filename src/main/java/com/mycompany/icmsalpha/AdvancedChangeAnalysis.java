package com.mycompany.icmsalpha;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Multi-threaded Change Analysis with concurrent processing at multiple levels
 * 
 * @author Administrator
 */
public class AdvancedChangeAnalysis {

    private static final Pattern DIFF_PATTERN = Pattern.compile("(\\d+(?:,\\d+)?)([acd])(\\d+(?:,\\d+)?)");
    private static final Pattern LINE_RANGE_PATTERN = Pattern.compile("(\\d+)(?:,(\\d+))?");

    // Thread pool configuration
    private final int revisionThreads;
    private final int fileThreads;
    private final ExecutorService revisionExecutor;
    private final ExecutorService fileExecutor;

    // Thread-safe counters
    private final AtomicInteger totalChanges = new AtomicInteger(0);
    private final AtomicInteger processedRevisions = new AtomicInteger(0);
    private final AtomicInteger processedFiles = new AtomicInteger(0);

    // Cache for file existence checks (thread-safe)
    private final Map<String, Boolean> fileExistsCache = new ConcurrentHashMap<>();

    private final AccessPoint ap;
    private final CommonParameters cp;
    private final LoadParam lp;

    // Single output file for all changes
    private Path outputFile;
    private BufferedWriter globalWriter;
    private final Object writerLock = new Object();
    private final AtomicInteger globalCounter = new AtomicInteger(1);

    public AdvancedChangeAnalysis() {
        int processors = Runtime.getRuntime().availableProcessors();
        this.revisionThreads = Math.max(2, processors / 2);
        this.fileThreads = Math.max(4, processors);

        this.revisionExecutor = Executors.newFixedThreadPool(
                revisionThreads,
                createThreadFactory("Revision-Worker"));

        this.fileExecutor = Executors.newFixedThreadPool(
                fileThreads,
                createThreadFactory("File-Worker"));

        // Initialize dependencies with error handling
        try {
            this.ap = new AccessPoint();
            this.cp = new CommonParameters();
            this.lp = new LoadParam();
            print("Initialized with " + revisionThreads + " revision threads and " +
                    fileThreads + " file processing threads");
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize dependencies: " + e.getMessage(), e);
        }
    }

    public AdvancedChangeAnalysis(int revisionThreads, int fileThreads) {
        this.revisionThreads = revisionThreads;
        this.fileThreads = fileThreads;

        this.revisionExecutor = Executors.newFixedThreadPool(
                revisionThreads,
                createThreadFactory("Revision-Worker"));

        this.fileExecutor = Executors.newFixedThreadPool(
                fileThreads,
                createThreadFactory("File-Worker"));

        // Initialize dependencies with error handling
        try {
            this.ap = new AccessPoint();
            this.cp = new CommonParameters();
            this.lp = new LoadParam();
            print("Initialized with " + revisionThreads + " revision threads and " +
                    fileThreads + " file processing threads");
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize dependencies: " + e.getMessage(), e);
        }
    }

    private ThreadFactory createThreadFactory(String namePrefix) {
        return new ThreadFactory() {
            private final AtomicInteger counter = new AtomicInteger(1);

            @Override
            public Thread newThread(Runnable r) {
                Thread thread = new Thread(r, namePrefix + "-" + counter.getAndIncrement());
                thread.setDaemon(true);
                return thread;
            }
        };
    }

    public void print(Object o) {
        System.out.println("[" + Thread.currentThread().getName() + "] " + o);
    }

    public void init() throws IOException, InterruptedException {
        long startTime = System.currentTimeMillis();

        try {
            List<SVNChanges> changeByRevision = ap.getSVNChanges();

            if (changeByRevision == null || changeByRevision.isEmpty()) {
                print("No SVN changes found.");
                return;
            }

            // Setup single output file
            setupOutputFile();

            // Write header
            synchronized (writerLock) {
                globalWriter.write("counter | prev_revision | prev_path | prev_lines | change_type | " +
                        "curr_revision | curr_path | curr_lines | fix_type\n");
                globalWriter.flush();
            }

            // Group by revision and sort efficiently
            Map<Integer, RevisionGroup> sortedChanges = changeByRevision.stream()
                    .collect(Collectors.groupingBy(
                            change -> change.revision,
                            TreeMap::new,
                            Collectors.collectingAndThen(
                                    Collectors.toList(),
                                    this::createRevisionGroup)));

            int totalRevisions = sortedChanges.size();
            print("Starting parallel analysis of " + totalRevisions + " revisions");
            print("Output file: " + outputFile.toAbsolutePath());

            // Get first revision separately
            Map.Entry<Integer, RevisionGroup> firstEntry = sortedChanges.entrySet().iterator().next();
            int firstRevision = firstEntry.getKey();
            RevisionGroup firstGroup = firstEntry.getValue();

            print("Processing FIRST revision " + firstRevision + " as ADD operations");
            processFirstRevisionAsAdd(firstRevision, firstGroup);

            // Submit remaining revision processing tasks
            List<Future<ProcessingResult>> futures = new ArrayList<>();

            for (Map.Entry<Integer, RevisionGroup> entry : sortedChanges.entrySet()) {
                int revision = entry.getKey();

                // Skip first revision (already processed) and revision 2
                if (revision == firstRevision || revision == 2 || revision >= lp.getLastRevNumber()) {
                    continue;
                }

                Future<ProcessingResult> future = revisionExecutor
                        .submit(() -> processRevisionParallel(revision, entry.getValue()));
                futures.add(future);
            }

            // Monitor progress in separate thread
            Thread progressMonitor = startProgressMonitor(futures.size());

            // Collect results
            int successCount = 1; // Count first revision
            int errorCount = 0;

            for (Future<ProcessingResult> future : futures) {
                try {
                    ProcessingResult result = future.get();
                    if (result.error != null) {
                        print("ERROR in revision " + result.revision + ": " + result.error);
                        errorCount++;
                    } else {
                        successCount++;
                    }
                } catch (ExecutionException e) {
                    print("Execution error: " + e.getMessage());
                    e.printStackTrace();
                    errorCount++;
                }
            }

            progressMonitor.interrupt();

            // Close the writer
            synchronized (writerLock) {
                if (globalWriter != null) {
                    globalWriter.close();
                    globalWriter = null;
                }
            }

            long duration = System.currentTimeMillis() - startTime;
            print("\n========== ANALYSIS COMPLETE ==========");
            print("Total time: " + (duration / 1000.0) + " seconds");
            print("Revisions processed: " + processedRevisions.get() + " (Success: " +
                    successCount + ", Errors: " + errorCount + ")");
            print("Files processed: " + processedFiles.get());
            print("Total changes detected: " + totalChanges.get());

            if (processedRevisions.get() > 0) {
                print("Throughput: " + String.format("%.2f", processedRevisions.get() / (duration / 1000.0)) +
                        " revisions/second");
            }

            print("\nOutput saved to:");
            print("  " + outputFile.toAbsolutePath());

            if (Files.exists(outputFile)) {
                print("  Size: " + String.format("%.2f KB", Files.size(outputFile) / 1024.0));
                print("  Lines: " + (globalCounter.get() - 1));
            }
            print("=======================================\n");

        } catch (Exception e) {
            print("FATAL ERROR: " + e.getMessage());
            e.printStackTrace();
            throw e;
        } finally {
            // Ensure writer is closed
            synchronized (writerLock) {
                if (globalWriter != null) {
                    try {
                        globalWriter.close();
                    } catch (IOException e) {
                        print("Error closing writer: " + e.getMessage());
                    }
                }
            }
            shutdownExecutors();
        }
    }

    /**
     * Process first revision - all files are treated as "add" operations
     */
    private void processFirstRevisionAsAdd(int revision, RevisionGroup group) {
        try {
            print("Processing first revision " + revision + " (" + group.filePaths.size() + " files)");

            Set<ChangeRecord> addedFiles = new LinkedHashSet<>();

            // For first revision, all files are additions
            for (String filePath : group.filePaths) {
                Path currPath = Paths.get(getFullPath(revision, filePath));

                // Check if file exists
                if (!fileExists(currPath)) {
                    continue;
                }

                // Count lines in the file
                int lineCount = 0;
                try {
                    lineCount = (int) Files.lines(currPath).count();
                } catch (IOException e) {
                    lineCount = 1; // Default to 1 if can't read
                }

                // Create an "add" record (no previous revision)
                ChangeRecord addRecord = new ChangeRecord(
                        0, // No previous revision
                        "N/A", // No previous path
                        new LineRange(0, 0), // No previous lines
                        'a', // Add operation
                        revision,
                        getRevPath(revision, filePath),
                        new LineRange(1, lineCount) // All lines are new
                );

                addedFiles.add(addRecord);
                processedFiles.incrementAndGet();
            }

            // Write all added files to output
            synchronized (writerLock) {
                if (globalWriter == null) {
                    throw new IOException("Global writer is null");
                }

                for (ChangeRecord change : addedFiles) {
                    int counter = globalCounter.getAndIncrement();
                    String outputLine = String.format("%d | %s | %s%n", counter, change.toString(), group.fixType);
                    globalWriter.write(outputLine);
                }
                globalWriter.flush();
            }

            int changeCount = addedFiles.size();
            totalChanges.addAndGet(changeCount);
            processedRevisions.incrementAndGet();

            print("First revision complete: " + changeCount + " files added");

        } catch (Exception e) {
            print("ERROR processing first revision: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void setupOutputFile() throws IOException {
        try {
            // Get repository name
            String repoName = extractRepoName();

            // Create output directory: workfolder/repo_name/Changes/
            String workFolder = cp.getCloneDir();

            if (workFolder == null || workFolder.trim().isEmpty()) {
                workFolder = System.getProperty("user.dir"); // Use current directory as fallback
                print("Warning: Clone directory not set, using current directory: " + workFolder);
            }

            Path changesDir = Paths.get(workFolder, repoName, "Changes");
            Files.createDirectories(changesDir);

            // Create output file: workfolder/repo_name/Changes/changes.txt
            // filepath

            //
            outputFile = changesDir.resolve("changes.txt");
            globalWriter = Files.newBufferedWriter(outputFile, StandardCharsets.UTF_8);

            print("Output directory created: " + changesDir.toAbsolutePath());
        } catch (Exception e) {
            print("ERROR setting up output file: " + e.getMessage());
            throw new IOException("Failed to setup output file", e);
        }
    }

    private String extractRepoName() {
        try {
            // Try to extract repo name from clone directory
            String cloneDir = cp.getCloneDir();
            if (cloneDir != null && !cloneDir.isEmpty()) {
                Path path = Paths.get(cloneDir);
                String lastName = path.getFileName().toString();
                if (lastName != null && !lastName.isEmpty()) {
                    return lastName;
                }
            }
        } catch (Exception e) {
            print("Warning: Could not extract repo name: " + e.getMessage());
        }
        return "repository"; // default fallback
    }

    private Thread startProgressMonitor(int totalTasks) {
        Thread monitor = new Thread(() -> {
            try {
                while (!Thread.interrupted()) {
                    int processed = processedRevisions.get();
                    int files = processedFiles.get();
                    int changes = totalChanges.get();
                    print(String.format("Progress: %d/%d revisions | %d files | %d changes",
                            processed, totalTasks, files, changes));
                    Thread.sleep(5000); // Update every 5 seconds
                }
            } catch (InterruptedException e) {
                // Exit gracefully
            }
        }, "Progress-Monitor");
        monitor.setDaemon(true);
        monitor.start();
        return monitor;
    }

    private RevisionGroup createRevisionGroup(List<SVNChanges> changes) {
        String fixType = changes.stream()
                .map(c -> c.fixType)
                .filter(Objects::nonNull)
                .findFirst()
                .orElse("UNKNOWN");

        List<String> filePaths = changes.stream()
                .map(c -> parseFilePath(c.filePath))
                .distinct()
                .collect(Collectors.toList());

        return new RevisionGroup(fixType, filePaths);
    }

    private ProcessingResult processRevisionParallel(int revision, RevisionGroup group) {
        ProcessingResult result = new ProcessingResult(revision);

        try {
            print("Starting revision " + revision + " with " + group.filePaths.size() + " files");
            result.changeCount = analyzeRevisionParallel(group.filePaths, revision, group.fixType);
            processedRevisions.incrementAndGet();
            print("Completed revision " + revision + " (" + result.changeCount + " changes)");
        } catch (Exception e) {
            result.error = e.getMessage();
            print("ERROR in revision " + revision + ": " + e.getMessage());
            e.printStackTrace();
        }

        return result;
    }

    private String parseFilePath(String fullPath) {
        if (fullPath == null)
            return "";

        try {
            int revisionIndex = fullPath.indexOf("Revision_");
            if (revisionIndex != -1) {
                int pathStartIndex = fullPath.indexOf('/', revisionIndex);
                if (pathStartIndex != -1) {
                    return fullPath.substring(pathStartIndex + 1);
                }
            }
        } catch (Exception e) {
            print("Warning: Could not parse file path: " + fullPath);
        }

        return fullPath;
    }

    int analyzeRevisionParallel(List<String> paths, int currRevision, String fixType)
            throws IOException, InterruptedException {
        // Thread-safe collection for changes
        ConcurrentLinkedQueue<ChangeRecord> allChanges = new ConcurrentLinkedQueue<>();

        // Process files in parallel
        List<Future<List<ChangeRecord>>> fileFutures = new ArrayList<>();

        for (String filePath : paths) {
            Future<List<ChangeRecord>> future = fileExecutor.submit(() -> processFileChange(currRevision, filePath));
            fileFutures.add(future);
        }

        // Collect all changes from parallel processing
        for (Future<List<ChangeRecord>> future : fileFutures) {
            try {
                List<ChangeRecord> changes = future.get();
                allChanges.addAll(changes);
                processedFiles.incrementAndGet();
            } catch (ExecutionException e) {
                print("Error processing file: " + e.getMessage());
            }
        }

        // Remove duplicates
        Set<ChangeRecord> uniqueChanges = new LinkedHashSet<>(allChanges);

        // Write to global file with synchronization
        synchronized (writerLock) {
            if (globalWriter == null) {
                throw new IOException("Global writer is null");
            }

            for (ChangeRecord change : uniqueChanges) {
                int counter = globalCounter.getAndIncrement();
                String outputLine = String.format("%d | %s | %s%n", counter, change.toString(), fixType);
                globalWriter.write(outputLine);
            }
            globalWriter.flush(); // Ensure data is written
        }

        int changeCount = uniqueChanges.size();
        totalChanges.addAndGet(changeCount);
        return changeCount;
    }

    private List<ChangeRecord> processFileChange(int currRevision, String filePath) {
        List<ChangeRecord> changes = new ArrayList<>();

        try {
            int prevRevision = currRevision - 1;
            Path currPath = Paths.get(getFullPath(currRevision, filePath));
            Path prevPath = Paths.get(getFullPath(prevRevision, filePath));

            // Skip if files don't exist
            if (!fileExists(currPath) || !fileExists(prevPath)) {
                return changes;
            }

            return executeDiff(prevPath, currPath, prevRevision, currRevision, filePath);
        } catch (Exception e) {
            print("Error in processFileChange: " + e.getMessage());
            return changes;
        }
    }

    private List<ChangeRecord> executeDiff(Path prevPath, Path currPath,
            int prevRevision, int currRevision,
            String filePath) {
        List<ChangeRecord> changes = new ArrayList<>();

        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "diff",
                    prevPath.toString(),
                    currPath.toString());
            pb.redirectErrorStream(true);
            Process process = pb.start();

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {

                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();

                    // Skip diff content lines
                    if (line.isEmpty() || line.startsWith("<") || line.startsWith(">") || line.startsWith("---")) {
                        continue;
                    }

                    // Parse diff header
                    ChangeRecord change = parseDiffLine(line, prevRevision, currRevision, filePath);
                    if (change != null) {
                        changes.add(change);
                    }
                }
            }

            // Wait for process with timeout
            if (!process.waitFor(30, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                print("Warning: Diff timeout for " + filePath);
            }

        } catch (IOException | InterruptedException e) {
            print("Error executing diff for " + filePath + ": " + e.getMessage());
        }

        return changes;
    }

    private ChangeRecord parseDiffLine(String line, int prevRevision, int currRevision, String filePath) {
        try {
            Matcher matcher = DIFF_PATTERN.matcher(line);

            if (!matcher.matches()) {
                return null;
            }

            String prevLineRange = matcher.group(1);
            char changeType = matcher.group(2).charAt(0);
            String currLineRange = matcher.group(3);

            LineRange prevRange = parseLineRange(prevLineRange);
            LineRange currRange = parseLineRange(currLineRange);

            return new ChangeRecord(
                    prevRevision,
                    getRevPath(prevRevision, filePath),
                    prevRange,
                    changeType,
                    currRevision,
                    getRevPath(currRevision, filePath),
                    currRange);
        } catch (Exception e) {
            print("Error parsing diff line: " + line + " - " + e.getMessage());
            return null;
        }
    }

    private LineRange parseLineRange(String rangeStr) {
        try {
            Matcher matcher = LINE_RANGE_PATTERN.matcher(rangeStr);
            if (matcher.matches()) {
                int start = Integer.parseInt(matcher.group(1));
                int end = matcher.group(2) != null ? Integer.parseInt(matcher.group(2)) : start;
                return new LineRange(start, end);
            }
        } catch (Exception e) {
            print("Error parsing line range: " + rangeStr);
        }
        return new LineRange(0, 0);
    }

    private boolean fileExists(Path path) {
        return fileExistsCache.computeIfAbsent(path.toString(), k -> Files.exists(path));
    }

    public String getRevPath(int revision, String filepath) {
        return "Revision_" + revision + "/" + filepath;
    }

    public String getFullPath(int revision, String filepath) {
        return cp.getCloneDir() + "/Revision_" + revision + "/" + filepath;
    }

    private void shutdownExecutors() {
        print("Shutting down thread pools...");

        shutdownExecutor(revisionExecutor, "Revision executor");
        shutdownExecutor(fileExecutor, "File executor");
    }

    private void shutdownExecutor(ExecutorService executor, String name) {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                print(name + " did not terminate in time, forcing shutdown");
                executor.shutdownNow();
                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                    print("ERROR: " + name + " did not terminate");
                }
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    // ==================== Inner Classes ====================

    private static class RevisionGroup {
        final String fixType;
        final List<String> filePaths;

        RevisionGroup(String fixType, List<String> filePaths) {
            this.fixType = fixType;
            this.filePaths = filePaths;
        }
    }

    private static class ProcessingResult {
        final int revision;
        int changeCount;
        String error;

        ProcessingResult(int revision) {
            this.revision = revision;
        }
    }

    private static class LineRange {
        final int start;
        final int end;

        LineRange(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return start == end ? String.valueOf(start) : start + "," + end;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o)
                return true;
            if (!(o instanceof LineRange))
                return false;
            LineRange lineRange = (LineRange) o;
            return start == lineRange.start && end == lineRange.end;
        }

        @Override
        public int hashCode() {
            return Objects.hash(start, end);
        }
    }

    private static class ChangeRecord {
        final int prevRevision;
        final String prevPath;
        final LineRange prevLines;
        final char changeType;
        final int currRevision;
        final String currPath;
        final LineRange currLines;

        ChangeRecord(int prevRevision, String prevPath, LineRange prevLines,
                char changeType, int currRevision, String currPath, LineRange currLines) {
            this.prevRevision = prevRevision;
            this.prevPath = prevPath;
            this.prevLines = prevLines;
            this.changeType = changeType;
            this.currRevision = currRevision;
            this.currPath = currPath;
            this.currLines = currLines;
        }

        @Override
        public String toString() {
            return String.format("%d | %s | %s | %c | %d | %s | %s",
                    prevRevision, prevPath, prevLines,
                    changeType, currRevision, currPath, currLines);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o)
                return true;
            if (!(o instanceof ChangeRecord))
                return false;
            ChangeRecord that = (ChangeRecord) o;
            return prevRevision == that.prevRevision &&
                    changeType == that.changeType &&
                    currRevision == that.currRevision &&
                    Objects.equals(prevPath, that.prevPath) &&
                    Objects.equals(prevLines, that.prevLines) &&
                    Objects.equals(currPath, that.currPath) &&
                    Objects.equals(currLines, that.currLines);
        }

        @Override
        public int hashCode() {
            return Objects.hash(prevRevision, prevPath, prevLines, changeType,
                    currRevision, currPath, currLines);
        }
    }

    // ==================== Main Method for Testing ====================

    public static void main(String[] args) {
        try {
            System.out.println("========== Starting Change Analysis ==========\n");

            // Option 1: Auto-configure based on CPU cores
            AdvancedChangeAnalysis analysis = new AdvancedChangeAnalysis();

            // Option 2: Manual thread configuration
            // AdvancedChangeAnalysis analysis = new AdvancedChangeAnalysis(4, 8);

            // Run the analysis
            analysis.init();

            System.out.println("\n========== Analysis Complete ==========");
            System.out.println("Check the output file at:");
            System.out.println("  workfolder/repo_name/Changes/changes.txt");

        } catch (Exception e) {
            System.err.println("Error during analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }
}