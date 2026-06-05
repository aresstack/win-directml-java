package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

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
    void initializesFromSafeTensorsByCompilingWdmlPackOnFirstUse() throws Exception {
        T5Config config = T5TestFixtures.tinyConfig(false);
        CodeT5TokenizerTest.writeTokenizerFiles(tempDir);
        T5TestFixtures.writeConfig(tempDir, config);
        Map<String, OnnxTensor> tensors = T5TestFixtures.completeDenseT5Tensors(config);
        T5TestFixtures.writeSafeTensors(tempDir, tensors);

        T5InferenceEngine engine = new T5InferenceEngine(tempDir, 2, 8);
        try {
            engine.initialize();

            assertTrue(engine.isReady());
            assertTrue(Files.isRegularFile(tempDir.resolve(T5InferenceEngine.DEFAULT_PACKAGE_NAME)));
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
    @Test
    void initializesFromTorchCheckpointByCompilingWdmlPackOnFirstUse() throws Exception {
        T5Config config = T5TestFixtures.tinyConfig(false);
        CodeT5TokenizerTest.writeTokenizerFiles(tempDir);
        T5TestFixtures.writeConfig(tempDir, config);
        T5TestFixtures.writeTorchCheckpoint(tempDir, T5TestFixtures.completeDenseT5Tensors(config));

        T5InferenceEngine engine = new T5InferenceEngine(tempDir, 2, 8);
        try {
            engine.initialize();

            assertTrue(engine.isReady());
            assertTrue(Files.isRegularFile(tempDir.resolve(T5InferenceEngine.DEFAULT_PACKAGE_NAME)));
            assertEquals("torch-state-dict", engine.runtimePackage().manifest().get("sourceFormat"));
        } finally {
            engine.shutdown();
        }
    }

}
