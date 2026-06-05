package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class T5WdmlPackCompileToolTest {

    @TempDir
    Path tempDir;

    @Test
    void dryRunPrintsLayoutReportWithoutWritingPackage() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("dry-run.wdmlpack");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();

        int exit = T5WdmlPackCompileTool.run(new String[]{"--dry-run", modelDir.toString(), output.toString()},
                new PrintStream(out), new PrintStream(err));

        assertEquals(0, exit, err.toString(StandardCharsets.UTF_8));
        assertFalse(Files.exists(output));
        String text = out.toString(StandardCharsets.UTF_8);
        assertTrue(text.contains("modelFamily=t5"));
        assertTrue(text.contains("runtimeLoadable=no"));
    }

    @Test
    void rejectsExistingOutputWithoutForce() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("existing.wdmlpack");
        Files.writeString(output, "already here");
        ByteArrayOutputStream err = new ByteArrayOutputStream();

        int exit = T5WdmlPackCompileTool.run(new String[]{modelDir.toString(), output.toString()},
                new PrintStream(new ByteArrayOutputStream()), new PrintStream(err));

        assertEquals(2, exit);
        assertTrue(err.toString(StandardCharsets.UTF_8).contains("--force"));
    }

    @Test
    void forceAllowsExistingOutput() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("existing.wdmlpack");
        Files.writeString(output, "already here");
        ByteArrayOutputStream err = new ByteArrayOutputStream();

        int exit = T5WdmlPackCompileTool.run(new String[]{"--force", modelDir.toString(), output.toString()},
                new PrintStream(new ByteArrayOutputStream()), new PrintStream(err));

        assertEquals(0, exit, err.toString(StandardCharsets.UTF_8));
        assertEquals("WDMLPACK", new String(Files.readAllBytes(output), 0, 8, StandardCharsets.US_ASCII));
    }

    @Test
    void inspectPrintsT5Manifest() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("inspect.wdmlpack");
        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();

        int exit = T5WdmlPackCompileTool.run(new String[]{"--inspect", output.toString()},
                new PrintStream(out), new PrintStream(err));

        assertEquals(0, exit, err.toString(StandardCharsets.UTF_8));
        assertTrue(out.toString(StandardCharsets.UTF_8).contains("modelFamily=t5"));
    }

    @Test
    void inspectRejectsNonT5Manifest() throws Exception {
        Path pack = tempDir.resolve("qwen.wdmlpack");
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("format", "wdmlpack");
        manifest.put("version", WdmlPackWriter.VERSION);
        manifest.put("modelFamily", "qwen2");
        manifest.put("runtimeLoadable", true);
        WdmlPackWriter.writeManifestOnly(pack, manifest);
        ByteArrayOutputStream err = new ByteArrayOutputStream();

        int exit = T5WdmlPackCompileTool.run(new String[]{"--inspect", pack.toString()},
                new PrintStream(new ByteArrayOutputStream()), new PrintStream(err));

        assertEquals(2, exit);
        assertTrue(err.toString(StandardCharsets.UTF_8).contains("Not a T5"));
    }

    private Path createModelDir(T5Config config) throws Exception {
        Path modelDir = tempDir.resolve("model-" + System.nanoTime());
        T5TestFixtures.writeConfig(modelDir, config);
        T5TestFixtures.writeSafeTensors(modelDir, T5TestFixtures.completeDenseT5Tensors(config));
        return modelDir;
    }
}
