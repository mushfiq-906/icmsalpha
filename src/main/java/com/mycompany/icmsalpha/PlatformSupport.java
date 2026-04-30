package com.mycompany.icmsalpha;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

final class PlatformSupport {
    private static final boolean WINDOWS = System.getProperty("os.name", "")
            .toLowerCase()
            .contains("win");
    private static final Path PROJECT_ROOT = Paths.get(System.getProperty("user.dir"))
            .toAbsolutePath()
            .normalize();

    private PlatformSupport() {
    }

    static boolean isWindows() {
        return WINDOWS;
    }

    static List<String> buildCtagsCommand(String language, String outputPath, String revisionPath) {
        List<String> command = new ArrayList<>();
        command.add(resolveCtagsExecutable());
        command.add("--fields=+n");
        command.add("-R");
        command.add("--languages=" + language);
        command.add("-o");
        command.add(outputPath);
        command.add(revisionPath);
        return command;
    }

    static ProcessBuilder createDiffProcessBuilder(String previousPath, String currentPath) {
        return new ProcessBuilder(resolveDiffExecutable(), previousPath, currentPath);
    }

    static String stripRevisionPrefix(String fullPath) {
        if (fullPath == null) {
            return "";
        }

        int revisionIndex = fullPath.indexOf("Revision_");
        if (revisionIndex == -1) {
            return fullPath;
        }

        int slashIndex = fullPath.indexOf('/', revisionIndex);
        int backslashIndex = fullPath.indexOf('\\', revisionIndex);

        int separatorIndex;
        if (slashIndex == -1) {
            separatorIndex = backslashIndex;
        } else if (backslashIndex == -1) {
            separatorIndex = slashIndex;
        } else {
            separatorIndex = Math.min(slashIndex, backslashIndex);
        }

        if (separatorIndex == -1) {
            return fullPath.substring(revisionIndex);
        }

        return fullPath.substring(separatorIndex + 1);
    }

    private static String resolveCtagsExecutable() {
        Path bundledPath = PROJECT_ROOT.resolve(Paths.get("Tools", "ctags", WINDOWS ? "ctags.exe" : "ctags"));
        return resolveExecutable(
                System.getenv("ICMSALPHA_CTAGS"),
                bundledPath,
                WINDOWS ? "ctags.exe" : "ctags",
                "ctags");
    }

    private static String resolveDiffExecutable() {
        Path bundledWindowsDiff = PROJECT_ROOT.resolve(Paths.get("Tools", "DiffWin", "bin", "diff.exe"));

        if (WINDOWS) {
            return resolveExecutable(
                    System.getenv("ICMSALPHA_DIFF"),
                    bundledWindowsDiff,
                    "diff.exe",
                    "diff");
        }

        return resolveExecutable(
                System.getenv("ICMSALPHA_DIFF"),
                null,
                "diff");
    }

    private static String resolveExecutable(String envOverride, Path bundledPath, String... fallbacks) {
        if (envOverride != null && !envOverride.isBlank()) {
            Path overriddenPath = Paths.get(envOverride.trim());
            if (Files.exists(overriddenPath)) {
                return overriddenPath.toAbsolutePath().toString();
            }
            return envOverride.trim();
        }

        if (bundledPath != null && Files.exists(bundledPath)) {
            return bundledPath.toAbsolutePath().toString();
        }

        for (String fallback : fallbacks) {
            if (fallback != null && !fallback.isBlank()) {
                return fallback;
            }
        }

        throw new IllegalStateException("No executable could be resolved for the requested tool.");
    }
}
