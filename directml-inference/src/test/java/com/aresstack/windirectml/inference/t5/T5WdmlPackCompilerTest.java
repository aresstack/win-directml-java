package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class T5WdmlPackCompilerTest {

    @TempDir
    Path tempDir;

    @Test
    void writesPayloadBackedPackage() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("t5.wdmlpack");

        T5WdmlPackCompiler.T5CompileResult result = T5WdmlPackCompiler.compile(
                new T5CompileOptions(modelDir, output, false, false));

        assertTrue(result.written());
        assertTrue(Files.isRegularFile(output));
        assertFalse(WdmlPackWriter.readHeader(output).manifestOnly());
        assertTrue(WdmlPackWriter.readHeader(output).payloadIncluded());
    }


    @Test
    void compilesPayloadPackageFromRestrictedTorchCheckpointWhenSafeTensorsIsAbsent() throws Exception {
        T5Config config = T5TestFixtures.tinyConfig(false);
        Path modelDir = tempDir.resolve("torch-model");
        T5TestFixtures.writeConfig(modelDir, config);
        T5TestFixtures.writeTorchCheckpoint(modelDir, T5TestFixtures.completeDenseT5Tensors(config));
        Path output = tempDir.resolve("torch-t5.wdmlpack");

        T5WdmlPackCompiler.T5CompileResult result = T5WdmlPackCompiler.compile(
                new T5CompileOptions(modelDir, output, false, false));

        assertTrue(result.written());
        assertTrue(Files.isRegularFile(output));
        Map<String, Object> manifest = WdmlPackWriter.readManifest(output);
        assertEquals("torch-state-dict", manifest.get("sourceFormat"));
        assertEquals(true, manifest.get("payloadIncluded"));
        assertEquals(true, manifest.get("weightsLoadable"));
        assertNotNull(result.runtimePackage());
    }

    @Test
    void marksPackageAsRuntimeLoadableAndExecutable() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("t5.wdmlpack");

        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));
        Map<String, Object> manifest = WdmlPackWriter.readManifest(output);

        // T5-1/T5-2: honest status — weights load, runtime builds structures, and the reference engine is verified
        // end-to-end, so the package is executable via the reference engine.
        assertEquals(true, manifest.get("payloadIncluded"));
        assertEquals(true, manifest.get("weightsLoadable"));
        assertEquals(true, manifest.get("runtimeLoadable"));
        assertEquals(true, manifest.get("executable"));
        assertEquals(T5ManifestPayloadPolicy.MODE_EXECUTABLE, manifest.get("runtimeLoadMode"));
    }

    @Test
    void includesT5Metadata() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("t5.wdmlpack");

        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));
        Map<String, Object> manifest = WdmlPackWriter.readManifest(output);

        assertEquals("t5", manifest.get("modelFamily"));
        @SuppressWarnings("unchecked")
        Map<String, Object> t5 = (Map<String, Object>) manifest.get("t5");
        assertEquals(4, t5.get("dModel"));
        assertEquals(2, t5.get("dKv"));
        assertEquals(8, t5.get("dFf"));
    }

    @Test
    void includesLayoutReport() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("t5.wdmlpack");

        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));
        Map<String, Object> manifest = WdmlPackWriter.readManifest(output);

        @SuppressWarnings("unchecked")
        Map<String, Object> layout = (Map<String, Object>) manifest.get("layout");
        assertEquals(true, layout.get("complete"));
        assertEquals(true, layout.get("runtimeLoadable"));
        assertEquals(27, layout.get("roleCount"));
    }

    private Path createModelDir(T5Config config) throws Exception {
        Path modelDir = tempDir.resolve("model-" + System.nanoTime());
        T5TestFixtures.writeConfig(modelDir, config);
        T5TestFixtures.writeSafeTensors(modelDir, T5TestFixtures.completeDenseT5Tensors(config));
        return modelDir;
    }
}
