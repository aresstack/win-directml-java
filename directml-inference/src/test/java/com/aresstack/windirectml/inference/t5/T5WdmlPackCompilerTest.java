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
    void writesManifestOnlyPackage() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("t5.wdmlpack");

        T5WdmlPackCompiler.T5CompileResult result = T5WdmlPackCompiler.compile(
                new T5CompileOptions(modelDir, output, false, false));

        assertTrue(result.written());
        assertTrue(Files.isRegularFile(output));
        assertTrue(WdmlPackWriter.readHeader(output).manifestOnly());
    }

    @Test
    void marksPackageAsNotRuntimeLoadable() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("t5.wdmlpack");

        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));
        Map<String, Object> manifest = WdmlPackWriter.readManifest(output);

        assertEquals(false, manifest.get("runtimeLoadable"));
        assertEquals(false, manifest.get("payloadIncluded"));
        assertEquals("not-implemented", manifest.get("runtimeLoadMode"));
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
        assertEquals(false, layout.get("runtimeLoadable"));
        assertEquals(27, layout.get("roleCount"));
    }

    private Path createModelDir(T5Config config) throws Exception {
        Path modelDir = tempDir.resolve("model-" + System.nanoTime());
        T5TestFixtures.writeConfig(modelDir, config);
        T5TestFixtures.writeSafeTensors(modelDir, T5TestFixtures.completeDenseT5Tensors(config));
        return modelDir;
    }
}
