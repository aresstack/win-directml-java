package com.aresstack.windirectml.catalog;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * Bytecode gate for the neutral catalog.
 *
 * <p>{@code directml-model-catalog} is consumed by AskAI's Java-8 host, so every compiled class must
 * be Java-8 bytecode (class-file major version 52). This test walks the module's own compiled main
 * classes and asserts the major version. It is written in Java-8-compatible source (no records / var
 * / switch expressions) to match the module's own constraint.
 */
class CatalogBytecodeVersionTest {

    /** JVM class-file major version for Java 8. */
    private static final int JAVA_8_MAJOR = 52;

    @Test
    void everyCatalogClassIsJava8Bytecode() throws Exception {
        Path classesRoot = mainClassesRoot();
        assertTrue(Files.isDirectory(classesRoot),
                "Expected compiled main classes directory, got: " + classesRoot);

        List<String> violations = new ArrayList<String>();
        int scanned = 0;
        Stream<Path> walk = Files.walk(classesRoot);
        try {
            List<Path> files = walk
                    .filter(new java.util.function.Predicate<Path>() {
                        public boolean test(Path p) {
                            return p.toString().endsWith(".class");
                        }
                    })
                    .collect(Collectors.toList());
            for (Path classFile : files) {
                scanned++;
                int major = majorVersion(classFile);
                if (major != JAVA_8_MAJOR) {
                    String relative = classesRoot.relativize(classFile).toString().replace('\\', '/');
                    violations.add(relative + " has bytecode major " + major
                            + " (expected " + JAVA_8_MAJOR + " for Java 8)");
                }
            }
        } finally {
            walk.close();
        }

        assertTrue(scanned > 0, "No compiled classes were scanned under " + classesRoot);
        assertEquals("[]", violations.toString(),
                "directml-model-catalog must stay Java-8 bytecode: " + violations);
    }

    private static Path mainClassesRoot() throws URISyntaxException {
        return Paths.get(
                LocalModelCatalog.class.getProtectionDomain().getCodeSource().getLocation().toURI());
    }

    private static int majorVersion(Path classFile) throws IOException {
        InputStream raw = Files.newInputStream(classFile);
        DataInputStream in = new DataInputStream(raw);
        try {
            int magic = in.readInt();
            if (magic != 0xCAFEBABE) {
                throw new IOException("Not a class file: " + classFile);
            }
            in.readUnsignedShort(); // minor
            return in.readUnsignedShort();
        } finally {
            in.close();
        }
    }
}
