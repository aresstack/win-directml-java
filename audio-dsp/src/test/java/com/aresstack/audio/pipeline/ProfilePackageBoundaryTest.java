package com.aresstack.audio.pipeline;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Guard the module's dependency direction: {@code pipeline} may depend on {@code profile}, but
 * {@code profile} must NEVER depend on {@code pipeline} (that package cycle is forbidden). This scans the
 * {@code profile} main sources and fails if any of them references the {@code pipeline} package.
 */
public class ProfilePackageBoundaryTest {

    private static final String FORBIDDEN = "com.aresstack.audio.pipeline";

    @Test
    public void profileSourcesDoNotDependOnPipeline() throws IOException {
        // Gradle runs the test with the module directory as the working directory.
        Path profileDir = Paths.get("src", "main", "java", "com", "aresstack", "audio", "profile");
        if (!Files.isDirectory(profileDir)) {
            fail("Cannot find the profile source directory at " + profileDir.toAbsolutePath()
                    + " — adjust the architecture test to the build layout.");
        }
        List<String> offenders = new ArrayList<String>();
        Stream<Path> files = Files.walk(profileDir);
        try {
            for (Path path : (Iterable<Path>) files::iterator) {
                if (!path.toString().endsWith(".java")) {
                    continue;
                }
                String content = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
                if (content.contains(FORBIDDEN)) {
                    offenders.add(path.toString());
                }
            }
        } finally {
            files.close();
        }
        assertTrue("profile sources must not reference " + FORBIDDEN + " (package cycle): " + offenders,
                offenders.isEmpty());
    }
}
