package com.aresstack.windirectml.inference.arch;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import com.aresstack.windirectml.inference.InferenceEngine;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * Architecture guard for the published generation runtime.
 *
 * <p>{@code directml-inference} is the swing-free, workbench-free, AskAI-free Java-21 generation
 * runtime that AskAI's sidecar and the Swing workbench both consume. This test parses the compiled
 * bytecode of every main class and fails if any of them reference a forbidden type, or if any class
 * is not Java-21 bytecode. It is dependency-free (no ArchUnit) so the published runtime keeps its
 * minimal test footprint.
 *
 * <p>Forbidden type references (internal/JVM names):
 * <ul>
 *   <li>{@code javax.swing}, {@code java.awt} — no UI toolkit in the runtime</li>
 *   <li>{@code com.aresstack.windirectml.workbench} — the runtime must not depend on the Swing
 *       workbench (the dependency direction is workbench &rarr; runtime, never the reverse)</li>
 *   <li>any {@code askai} package — the runtime is AskAI-agnostic; AskAI maps the neutral contract</li>
 *   <li>{@code java.lang.ProcessBuilder}, {@code java.lang.Runtime} — no external process / Python
 *       runner may be launched from the runtime</li>
 *   <li>{@code ai.onnxruntime}, {@code com.microsoft.onnxruntime} — ONNX is an import weight
 *       container only; the runtime never links ONNX Runtime</li>
 * </ul>
 *
 * <p>Only {@code CONSTANT_Class} entries are inspected (real type references), never string
 * literals, so descriptive labels such as {@code "external-python-transformers"} do not trip it.
 */
class RuntimeArchitectureTest {

    /** JVM class-file major version for Java 21. */
    private static final int JAVA_21_MAJOR = 65;

    /** Forbidden {@code CONSTANT_Class} internal-name prefixes. */
    private static final String[] FORBIDDEN_PREFIXES = {
        "javax/swing/",
        "java/awt/",
        "com/aresstack/windirectml/workbench/",
        "ai/onnxruntime/",
        "com/microsoft/onnxruntime/",
    };

    /**
     * Forbidden {@code CONSTANT_Class} exact internal names. {@code ProcessBuilder} is the external
     * process / Python-runner mechanism and is never benign in the runtime. {@code java.lang.Runtime}
     * itself is <em>allowed</em> (used for {@code availableProcessors()} / memory queries); only the
     * process-launching {@code Runtime.exec(...)} call is forbidden — detected via method references.
     */
    private static final String[] FORBIDDEN_EXACT = {
        "java/lang/ProcessBuilder",
    };

    /** Forbidden substring (any package segment named {@code askai}). */
    private static final String FORBIDDEN_ASKAI_SEGMENT = "/askai/";

    @Test
    void runtimeIsSwingWorkbenchAndAskAiFreeAndJava21() throws Exception {
        Path classesRoot = mainClassesRoot();
        assertTrue(Files.isDirectory(classesRoot),
                "Expected compiled main classes directory, got: " + classesRoot);

        List<String> violations = new ArrayList<>();
        int scanned = 0;
        Map<Integer, Integer> majorHistogram = new TreeMap<>();

        try (Stream<Path> classFiles = Files.walk(classesRoot)) {
            List<Path> files = classFiles
                    .filter(p -> p.toString().endsWith(".class"))
                    .collect(Collectors.toList());
            for (Path classFile : files) {
                scanned++;
                ClassInfo info = parse(classFile);
                majorHistogram.merge(info.major, 1, Integer::sum);

                String relative = classesRoot.relativize(classFile).toString().replace('\\', '/');
                if (info.major != JAVA_21_MAJOR) {
                    violations.add(relative + " has bytecode major " + info.major
                            + " (expected " + JAVA_21_MAJOR + " for Java 21)");
                }
                for (String referenced : info.classRefs) {
                    String bad = forbiddenReason(referenced);
                    if (bad != null) {
                        violations.add(relative + " references forbidden type "
                                + referenced.replace('/', '.') + " (" + bad + ")");
                    }
                }
                if (info.callsRuntimeExec) {
                    violations.add(relative + " calls java.lang.Runtime.exec(...) "
                            + "(no external process / Python runner)");
                }
            }
        }

        assertTrue(scanned > 0, "No compiled classes were scanned under " + classesRoot);
        if (!violations.isEmpty()) {
            fail("directml-inference architecture violations (" + violations.size() + "):\n  "
                    + String.join("\n  ", violations));
        }
    }

    private static String forbiddenReason(String internalName) {
        // Normalise array descriptors like "[Ljavax/swing/JFrame;" down to the element type.
        String name = internalName;
        while (name.startsWith("[")) {
            name = name.substring(1);
        }
        if (name.startsWith("L") && name.endsWith(";")) {
            name = name.substring(1, name.length() - 1);
        }
        for (String prefix : FORBIDDEN_PREFIXES) {
            if (name.startsWith(prefix)) {
                return "forbidden package";
            }
        }
        for (String exact : FORBIDDEN_EXACT) {
            if (name.equals(exact)) {
                return "no external process / Python runner";
            }
        }
        if (name.contains(FORBIDDEN_ASKAI_SEGMENT) || name.startsWith("askai/")) {
            return "runtime must stay AskAI-agnostic";
        }
        return null;
    }

    /** Locate the compiled main classes directory via a stable anchor class. */
    private static Path mainClassesRoot() throws URISyntaxException {
        Path location = Paths.get(
                InferenceEngine.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        // Under Gradle test execution this is build/classes/java/main (a directory).
        return location;
    }

    private static ClassInfo parse(Path classFile) throws IOException {
        try (InputStream raw = Files.newInputStream(classFile);
                DataInputStream in = new DataInputStream(raw)) {
            int magic = in.readInt();
            if (magic != 0xCAFEBABE) {
                throw new IOException("Not a class file: " + classFile);
            }
            in.readUnsignedShort(); // minor
            int major = in.readUnsignedShort();
            int cpCount = in.readUnsignedShort();

            String[] utf8 = new String[cpCount];
            int[] classNameIndex = new int[cpCount];
            int[] methodrefClassIndex = new int[cpCount];
            int[] methodrefNatIndex = new int[cpCount];
            int[] natNameIndex = new int[cpCount];
            for (int i = 1; i < cpCount; i++) {
                int tag = in.readUnsignedByte();
                switch (tag) {
                    case 1: // Utf8
                        utf8[i] = in.readUTF();
                        break;
                    case 7: // Class
                        classNameIndex[i] = in.readUnsignedShort();
                        break;
                    case 8: // String
                    case 16: // MethodType
                    case 19: // Module
                    case 20: // Package
                        in.readUnsignedShort();
                        break;
                    case 15: // MethodHandle
                        in.readUnsignedByte();
                        in.readUnsignedShort();
                        break;
                    case 10: // Methodref
                    case 11: // InterfaceMethodref
                        methodrefClassIndex[i] = in.readUnsignedShort();
                        methodrefNatIndex[i] = in.readUnsignedShort();
                        break;
                    case 12: // NameAndType
                        natNameIndex[i] = in.readUnsignedShort();
                        in.readUnsignedShort(); // descriptor index
                        break;
                    case 3: // Integer
                    case 4: // Float
                    case 9: // Fieldref
                    case 17: // Dynamic
                    case 18: // InvokeDynamic
                        in.readInt();
                        break;
                    case 5: // Long
                    case 6: // Double
                        in.readLong();
                        i++; // 8-byte constants occupy two pool slots
                        break;
                    default:
                        throw new IOException("Unknown constant pool tag " + tag + " in " + classFile);
                }
            }

            List<String> classRefs = new ArrayList<>();
            for (int i = 1; i < cpCount; i++) {
                if (classNameIndex[i] != 0) {
                    String name = utf8[classNameIndex[i]];
                    if (name != null) {
                        classRefs.add(name);
                    }
                }
            }

            boolean callsRuntimeExec = false;
            for (int i = 1; i < cpCount && !callsRuntimeExec; i++) {
                if (methodrefClassIndex[i] == 0) {
                    continue;
                }
                String owner = utf8[classNameIndex[methodrefClassIndex[i]]];
                String method = utf8[natNameIndex[methodrefNatIndex[i]]];
                if ("java/lang/Runtime".equals(owner) && "exec".equals(method)) {
                    callsRuntimeExec = true;
                }
            }
            return new ClassInfo(major, classRefs, callsRuntimeExec);
        }
    }

    private static final class ClassInfo {
        final int major;
        final List<String> classRefs;
        final boolean callsRuntimeExec;

        ClassInfo(int major, List<String> classRefs, boolean callsRuntimeExec) {
            this.major = major;
            this.classRefs = classRefs;
            this.callsRuntimeExec = callsRuntimeExec;
        }
    }

    // Keep an explicit reference so the anchor import is never optimised away by static analysis.
    static {
        Map<String, Class<?>> anchor = new LinkedHashMap<>();
        anchor.put("engine", InferenceEngine.class);
    }
}
