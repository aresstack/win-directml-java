package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class T5InferenceEngineTest {
    @TempDir
    Path tempDir;

    @Test
    void describesMissingRuntimePackageOrSafeTensors() throws Exception {
        CodeT5TokenizerTest.writeTokenizerFiles(tempDir);
        T5TestFixtures.writeConfig(tempDir, T5TestFixtures.tinyConfig(false));

        String missing = T5InferenceEngine.describeMissingModelFile(tempDir);

        assertNotNull(missing);
        assertTrue(missing.contains("Missing T5 runtime package"));
    }

    @Test
    void acceptsTorchCheckpointAsAutoCompileSource() throws Exception {
        T5Config config = T5TestFixtures.tinyConfig(false);
        CodeT5TokenizerTest.writeTokenizerFiles(tempDir);
        T5TestFixtures.writeConfig(tempDir, config);
        T5TestFixtures.writeTorchCheckpoint(tempDir, T5TestFixtures.completeDenseT5Tensors(config));

        String missing = T5InferenceEngine.describeMissingModelFile(tempDir);

        assertNull(missing);
    }

    @Test
    void acceptsGoogleT5TokenizerJsonAsTokenizerSource() throws Exception {
        JsonT5TokenizerTest.writeTokenizerFiles(tempDir);
        T5TestFixtures.writeConfig(tempDir, T5TestFixtures.tinyConfig(false));

        String missing = T5InferenceEngine.describeMissingModelFile(tempDir);

        assertNotNull(missing);
        assertTrue(missing.contains("Missing T5 runtime package"));
    }

    @Test
    void initializeWithoutPackageFailsFastAndNeverCompiles() throws Exception {
        // W2: the runtime must never compile. A SafeTensors-only directory (no .wdmlpack) must fail
        // fast with an actionable error and must not produce a package as a side effect of init.
        T5Config config = T5TestFixtures.tinyConfig(false);
        CodeT5TokenizerTest.writeTokenizerFiles(tempDir);
        T5TestFixtures.writeConfig(tempDir, config);
        T5TestFixtures.writeSafeTensors(tempDir, T5TestFixtures.completeDenseT5Tensors(config));

        T5InferenceEngine engine = new T5InferenceEngine(tempDir, 2, 8);
        try {
            InferenceException error = assertThrows(InferenceException.class, engine::initialize);
            assertTrue(error.getMessage().contains("Download tab -> Convert"),
                    "actionable error must point at the Convert flow: " + error.getMessage());
            assertFalse(engine.isReady());
            assertFalse(Files.isRegularFile(tempDir.resolve(T5InferenceEngine.DEFAULT_PACKAGE_NAME)),
                    "init must not implicitly compile a wdmlpack");
        } finally {
            engine.shutdown();
        }
    }

    @Test
    void initializesAndGeneratesFromPrebuiltPackageWithoutRecompiling() throws Exception {
        // Build the package up front (the only allowed write path), then remove the SafeTensors source.
        // If init/generate tried to compile, they would fail for lack of a source - so success here
        // proves the engine loads only the existing package and never compiles.
        T5Config config = T5TestFixtures.tinyConfig(false);
        CodeT5TokenizerTest.writeTokenizerFiles(tempDir);
        T5TestFixtures.writeConfig(tempDir, config);
        T5TestFixtures.writeSafeTensors(tempDir, T5TestFixtures.completeDenseT5Tensors(config));
        T5WdmlPackCompiler.compile(new T5CompileOptions(
                tempDir, tempDir.resolve(T5InferenceEngine.DEFAULT_PACKAGE_NAME), false, true));
        Files.delete(tempDir.resolve("model.safetensors"));

        T5InferenceEngine engine = new T5InferenceEngine(tempDir, 2, 8);
        try {
            engine.initialize();

            assertTrue(engine.isReady());
            InferenceResult result = engine.generate(InferenceRequest.builder()
                    .modelId("Salesforce/codet5-small")
                    .userPrompt("Hi")
                    .maxTokens(2)
                    .temperature(0.0f)
                    .build());
            assertNotNull(result.getText());
            assertNotNull(result.getUsage());
            assertTrue(result.getUsage().completionTokens() > 0);
        } finally {
            engine.shutdown();
        }
    }

}
