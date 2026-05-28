package com.aresstack.windirectml.workbench.launcher;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Java 8 bootstrapper that discovers a Java 21+ JVM and launches the
 * DirectML Workbench jar as a separate process.
 *
 * <p>This class is compiled with {@code --release 8} so it can run on any
 * Java version the user may have as default. It then locates a Java 21 or
 * newer installation to start the real Workbench application.
 *
 * <h3>Java discovery order</h3>
 * <ol>
 *   <li>{@code JAVA_HOME_21_X64} environment variable</li>
 *   <li>{@code JAVA_HOME} environment variable</li>
 *   <li>Every {@code java} executable found on {@code PATH}</li>
 *   <li>Common Windows installation locations as fallback</li>
 * </ol>
 */
public final class WorkbenchLauncher {

    /** Minimum Java major version required by the Workbench. */
    static final int MIN_JAVA_VERSION = 21;

    /** Pattern matching the version string in {@code java -version} output. */
    static final Pattern VERSION_PATTERN = Pattern.compile("version \"([^\"]+)\"");

    private WorkbenchLauncher() {
    }

    public static void main(String[] args) {
        File workbenchJar = resolveWorkbenchJar();
        if (workbenchJar == null) {
            System.err.println("[Launcher] ERROR: Could not find directml-workbench-all.jar.");
            System.err.println("  Place the Workbench jar next to the launcher jar, or set the");
            System.err.println("  WORKBENCH_JAR environment variable to its absolute path.");
            System.exit(1);
        }

        String javaExe = findJava21OrNewer();
        if (javaExe == null) {
            System.err.println("[Launcher] ERROR: Java " + MIN_JAVA_VERSION + " or newer was not found.");
            System.err.println();
            System.err.println("  The DirectML Workbench requires Java " + MIN_JAVA_VERSION + "+.");
            System.err.println();
            System.err.println("  To fix this, do one of the following:");
            System.err.println("    1. Install Java 21 (e.g. from https://adoptium.net/)");
            System.err.println("    2. Set JAVA_HOME_21_X64 to the JDK root directory");
            System.err.println("    3. Set JAVA_HOME to a Java 21+ installation");
            System.err.println("    4. Add a Java 21+ bin directory to your PATH");
            System.exit(1);
        }

        List<String> command = new ArrayList<String>();
        command.add(javaExe);
        command.add("--enable-preview");
        command.add("--enable-native-access=ALL-UNNAMED");
        command.add("--add-modules=jdk.incubator.vector");
        command.add("-jar");
        command.add(workbenchJar.getAbsolutePath());
        for (String arg : args) {
            command.add(arg);
        }

        System.out.println("[Launcher] Starting Workbench with: " + javaExe);
        System.out.println("[Launcher] Jar: " + workbenchJar.getAbsolutePath());

        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.inheritIO();
            Process process = pb.start();
            int exitCode = process.waitFor();
            System.exit(exitCode);
        } catch (Exception e) {
            System.err.println("[Launcher] ERROR: Failed to start Workbench process.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Resolves the Workbench fat jar location. Checks:
     * <ol>
     *   <li>WORKBENCH_JAR environment variable</li>
     *   <li>Same directory as the launcher jar</li>
     * </ol>
     */
    static File resolveWorkbenchJar() {
        String envJar = System.getenv("WORKBENCH_JAR");
        if (envJar != null && !envJar.isEmpty()) {
            File f = new File(envJar);
            if (f.isFile()) {
                return f;
            }
        }

        // Locate directory containing the launcher jar
        File launcherDir = getLauncherDirectory();
        if (launcherDir != null) {
            File candidate = new File(launcherDir, "directml-workbench-all.jar");
            if (candidate.isFile()) {
                return candidate;
            }
        }

        return null;
    }

    /**
     * Returns the directory containing the launcher jar, or null if it
     * cannot be determined.
     */
    static File getLauncherDirectory() {
        try {
            File jarFile = new File(
                    WorkbenchLauncher.class.getProtectionDomain()
                            .getCodeSource().getLocation().toURI());
            if (jarFile.isFile()) {
                return jarFile.getParentFile();
            }
            // If running from a classes directory (e.g. during development)
            return jarFile;
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Discovers a Java 21+ executable using the documented search order.
     *
     * @return absolute path to a suitable java executable, or null
     */
    static String findJava21OrNewer() {
        List<String> candidates = buildCandidateList();
        Set<String> seen = new LinkedHashSet<String>();

        for (String candidate : candidates) {
            if (candidate == null || candidate.isEmpty()) {
                continue;
            }
            // Normalize path for deduplication
            File f = new File(candidate);
            String canonical;
            try {
                canonical = f.getCanonicalPath();
            } catch (Exception e) {
                canonical = f.getAbsolutePath();
            }
            if (seen.contains(canonical)) {
                continue;
            }
            seen.add(canonical);

            if (!f.exists() || !f.canExecute()) {
                continue;
            }

            int major = getJavaMajorVersion(candidate);
            if (major >= MIN_JAVA_VERSION) {
                System.out.println("[Launcher] Found Java " + major + " at: " + candidate);
                return candidate;
            } else if (major > 0) {
                System.out.println("[Launcher] Skipping Java " + major + " at: " + candidate);
            }
        }

        return null;
    }

    /**
     * Builds an ordered list of java executable candidates.
     */
    static List<String> buildCandidateList() {
        List<String> candidates = new ArrayList<String>();
        String sep = File.separator;
        String exe = isWindows() ? "java.exe" : "java";

        // 1. JAVA_HOME_21_X64
        String jh21 = System.getenv("JAVA_HOME_21_X64");
        if (jh21 != null && !jh21.isEmpty()) {
            candidates.add(jh21 + sep + "bin" + sep + exe);
        }

        // 2. JAVA_HOME
        String jh = System.getenv("JAVA_HOME");
        if (jh != null && !jh.isEmpty()) {
            candidates.add(jh + sep + "bin" + sep + exe);
        }

        // 3. Every java on PATH
        String path = System.getenv("PATH");
        if (path == null) {
            path = System.getenv("Path");
        }
        if (path != null) {
            String[] dirs = path.split(Pattern.quote(File.pathSeparator));
            for (String dir : dirs) {
                if (dir.isEmpty()) {
                    continue;
                }
                File javaFile = new File(dir, exe);
                if (javaFile.exists()) {
                    candidates.add(javaFile.getAbsolutePath());
                }
            }
        }

        // 4. Common Windows installation locations
        if (isWindows()) {
            String programFiles = System.getenv("ProgramFiles");
            if (programFiles == null) {
                programFiles = "C:\\Program Files";
            }
            addWindowsFallbackCandidates(candidates, programFiles, exe);

            String programFilesX86 = System.getenv("ProgramFiles(x86)");
            if (programFilesX86 != null) {
                addWindowsFallbackCandidates(candidates, programFilesX86, exe);
            }
        }

        return candidates;
    }

    private static void addWindowsFallbackCandidates(List<String> candidates, String root, String exe) {
        String sep = File.separator;
        // Eclipse Adoptium / Temurin
        File adoptiumDir = new File(root, "Eclipse Adoptium");
        if (adoptiumDir.isDirectory()) {
            File[] jdks = adoptiumDir.listFiles();
            if (jdks != null) {
                for (File jdk : jdks) {
                    candidates.add(jdk.getAbsolutePath() + sep + "bin" + sep + exe);
                }
            }
        }
        // Common JDK directory names
        String[] prefixes = {"Java", "Microsoft", "Eclipse Foundation"};
        for (String prefix : prefixes) {
            File dir = new File(root, prefix);
            if (dir.isDirectory()) {
                File[] entries = dir.listFiles();
                if (entries != null) {
                    for (File entry : entries) {
                        File bin = new File(entry, "bin" + sep + exe);
                        if (bin.exists()) {
                            candidates.add(bin.getAbsolutePath());
                        }
                    }
                }
            }
        }
    }

    /**
     * Runs {@code java -version} and parses the major version number.
     *
     * @return the major version (e.g. 21), or -1 if parsing fails
     */
    static int getJavaMajorVersion(String javaPath) {
        try {
            ProcessBuilder pb = new ProcessBuilder(javaPath, "-version");
            pb.redirectErrorStream(true);
            Process proc = pb.start();
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(proc.getInputStream()));
            String line;
            int major = -1;
            while ((line = reader.readLine()) != null) {
                major = parseMajorVersion(line);
                if (major > 0) {
                    break;
                }
            }
            proc.waitFor();
            reader.close();
            return major;
        } catch (Exception e) {
            return -1;
        }
    }

    /**
     * Parses a major version number from a {@code java -version} output line.
     *
     * <p>Handles both old-style ({@code "1.8.0_xxx"}) and new-style
     * ({@code "21.0.1"}) version strings.
     *
     * @return the major version, or -1 if the line doesn't contain a version
     */
    static int parseMajorVersion(String line) {
        if (line == null) {
            return -1;
        }
        Matcher m = VERSION_PATTERN.matcher(line);
        if (!m.find()) {
            return -1;
        }
        String version = m.group(1);
        return extractMajor(version);
    }

    /**
     * Extracts the major version number from a version string.
     *
     * @param version e.g. "21.0.1", "1.8.0_392", "17"
     * @return the major version number
     */
    static int extractMajor(String version) {
        if (version == null || version.isEmpty()) {
            return -1;
        }
        // Strip any leading/trailing whitespace
        version = version.trim();

        // Split on dots
        String[] parts = version.split("[.+\\-]");
        if (parts.length == 0) {
            return -1;
        }

        try {
            int first = Integer.parseInt(parts[0]);
            // Old-style versioning: 1.8.0 means Java 8
            if (first == 1 && parts.length > 1) {
                return Integer.parseInt(parts[1]);
            }
            return first;
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    static boolean isWindows() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("win");
    }
}
