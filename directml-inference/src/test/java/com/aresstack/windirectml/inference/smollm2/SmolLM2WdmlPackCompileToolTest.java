package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2WdmlPackCompileToolTest {

    @TempDir
    Path tempDir;

    @Test
    void printsHelp() {
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();

        int exitCode = SmolLM2WdmlPackCompileTool.run(new String[]{"--help"},
                new PrintStream(stdout), new PrintStream(new ByteArrayOutputStream()));

        assertEquals(0, exitCode);
        assertTrue(stdout.toString().contains("--model-dir"));
    }

    @Test
    void dryRunDoesNotWriteOutput() throws Exception {
        SmolLM2TestFixtures.writeModelDirectory(tempDir, SmolLM2TestFixtures.config135(false), true);
        Path output = tempDir.resolve("smol.wdmlpack");
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();

        int exitCode = SmolLM2WdmlPackCompileTool.run(new String[]{
                "--model-dir", tempDir.toString(),
                "--output", output.toString(),
                "--dry-run"
        }, new PrintStream(stdout), new PrintStream(new ByteArrayOutputStream()));

        assertEquals(0, exitCode);
        assertFalse(Files.exists(output));
        String text = stdout.toString();
        assertTrue(text.contains("layoutComplete=yes"));
        assertTrue(text.contains("runtimeLoadableReason=SmolLM2 runtime is not implemented yet"));
    }

    @Test
    void reportsMissingConfigJson() {
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();

        int exitCode = SmolLM2WdmlPackCompileTool.run(new String[]{"--model-dir", tempDir.toString(), "--dry-run"},
                new PrintStream(new ByteArrayOutputStream()), new PrintStream(stderr));

        assertEquals(2, exitCode);
        assertTrue(stderr.toString().contains("missing config.json"));
    }

    @Test
    void rejectsUnsupportedModelType() throws Exception {
        SmolLM2Config unsupported = new SmolLM2Config("qwen2", java.util.List.of("LlamaForCausalLM"),
                8, 16, 1, 4, 2, 2, 32, 64, 1.0e-5d, 10000.0d,
                "silu", false, false, 1, 2, null, true);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, unsupported, true);
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();

        int exitCode = SmolLM2WdmlPackCompileTool.run(new String[]{"--model-dir", tempDir.toString(), "--dry-run"},
                new PrintStream(new ByteArrayOutputStream()), new PrintStream(stderr));

        assertEquals(2, exitCode);
        assertTrue(stderr.toString().contains("unsupported model_type"));
    }

    @Test
    void refusesToOverwriteWithoutForce() throws Exception {
        SmolLM2TestFixtures.writeModelDirectory(tempDir, SmolLM2TestFixtures.config135(false), true);
        Path output = tempDir.resolve("existing.wdmlpack");
        Files.writeString(output, "existing");
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();

        int exitCode = SmolLM2WdmlPackCompileTool.run(new String[]{
                "--model-dir", tempDir.toString(),
                "--output", output.toString()
        }, new PrintStream(new ByteArrayOutputStream()), new PrintStream(stderr));

        assertEquals(2, exitCode);
        assertTrue(stderr.toString().contains("output already exists"));
    }

    @Test
    void reportsRuntimeNotImplemented() throws Exception {
        SmolLM2TestFixtures.writeModelDirectory(tempDir, SmolLM2TestFixtures.config135(false), true);
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();

        int exitCode = SmolLM2WdmlPackCompileTool.run(new String[]{"--model-dir", tempDir.toString(), "--dry-run"},
                new PrintStream(stdout), new PrintStream(new ByteArrayOutputStream()));

        assertEquals(0, exitCode);
        assertTrue(stdout.toString().contains("runtimeLoadable=no"));
    }

    @Test
    void writesOutputWithForce() throws Exception {
        SmolLM2TestFixtures.writeModelDirectory(tempDir, SmolLM2TestFixtures.config135(false), true);
        Path output = tempDir.resolve("manifest.wdmlpack");
        Files.writeString(output, "existing");

        int exitCode = SmolLM2WdmlPackCompileTool.run(new String[]{
                "--model-dir", tempDir.toString(),
                "--output", output.toString(),
                "--force"
        }, new PrintStream(new ByteArrayOutputStream()), new PrintStream(new ByteArrayOutputStream()));

        assertEquals(0, exitCode);
        assertTrue(Files.size(output) > 64);
    }
}
