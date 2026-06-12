package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class T5RuntimeTest {

    @TempDir
    Path tempDir;

    @Test
    void loadsT5RuntimePackageMetadata() throws Exception {
        Path pack = writePack("t5", "encoder-decoder");

        T5RuntimePackage runtimePackage = T5RuntimePackage.open(pack);

        assertEquals(4, runtimePackage.metadata().dModel());
        assertFalse(runtimePackage.runtimeLoadable());
    }

    @Test
    void rejectsNonT5PackageMetadata() throws Exception {
        Path pack = writePack("qwen2", "decoder-only");

        Exception error = assertThrows(Exception.class, () -> T5RuntimePackage.open(pack));

        assertTrue(error.getMessage().contains("Not a T5"));
    }

    @Test
    void loadsPayloadBackedWeightsFromCompiledPackage() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("runtime-loadable-weights.wdmlpack");
        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));

        T5RuntimePackage runtimePackage = T5RuntimePackage.open(output);
        T5Weights weights = runtimePackage.weights();

        assertTrue(runtimePackage.payloadIncluded());
        assertTrue(runtimePackage.weightsLoadable());
        // T5-1: weights-loadable package is now honestly runtime-loadable (loader builds weights + structures),
        // but generation is not yet certified -> not executable.
        assertTrue(runtimePackage.runtimeLoadable());
        assertFalse(runtimePackage.executable());
        assertEquals(27, weights.tensorCount());
        assertTrue(weights.payloadBytes() > 0);
        assertArrayEquals(new long[]{6, 4}, weights.sharedEmbedding().dims());
        assertSame(weights.sharedEmbedding(), weights.lmHead());
        assertArrayEquals(new long[]{4, 4}, weights.encoderSelfAttention(0, "q").dims());
        assertArrayEquals(new long[]{4, 4}, weights.decoderCrossAttention(0, "o").dims());
    }

    @Test
    void runtimeLoadsWeightsAndExposesReferenceGenerationLoop() throws Exception {
        Path modelDir = createModelDir(T5TestFixtures.tinyConfig(false));
        Path output = tempDir.resolve("runtime-shell.wdmlpack");
        T5WdmlPackCompiler.compile(new T5CompileOptions(modelDir, output, false, false));
        T5RuntimePackage runtimePackage = T5RuntimePackage.open(output);
        T5Runtime runtime = T5Runtime.load(runtimePackage);

        assertEquals("reference", runtime.executionMode());
        assertSame(runtime.encoderPipeline(), runtime.encoderRunner());
        assertSame(runtime.decoderPipeline(), runtime.decoderRunner());
        assertEquals("reference-encoder", runtime.encoderRunner().executionMode());
        assertEquals("reference-decoder", runtime.decoderRunner().executionMode());

        T5RuntimeResult result = runtime.generate(T5RuntimeRequest.greedy(
                new int[]{1, 2}, 2, T5TestFixtures.tinyConfig(false).specialTokens()));

        assertEquals(27, runtime.weights().tensorCount());
        assertEquals(2, result.generatedTokens());
        assertEquals(T5RuntimeResult.FinishReason.max_tokens, result.finishReason());
        assertTrue(result.generationMetrics().runtimeNanos() > 0L);
        assertTrue(result.generationMetrics().encoderNanos() > 0L);
        assertFalse(result.generationMetrics().diagnosticLines().isEmpty());
    }

    private Path writePack(String family, String architecture) throws Exception {
        Path pack = tempDir.resolve(family + ".wdmlpack");
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("format", "wdmlpack");
        manifest.put("version", WdmlPackWriter.VERSION);
        manifest.put("modelFamily", family);
        manifest.put("architecture", architecture);
        manifest.put("runtimeLoadable", false);
        manifest.put("weightsLoadable", false);
        manifest.put("payloadIncluded", false);
        manifest.put("runtimeLoadMode", T5ManifestPayloadPolicy.MODE_MANIFEST_ONLY);
        manifest.put("t5", T5PackageMetadata.from(T5TestFixtures.tinyConfig(false)).toManifest());
        WdmlPackWriter.writeManifestOnly(pack, manifest);
        return pack;
    }

    private Path createModelDir(T5Config config) throws Exception {
        Path modelDir = tempDir.resolve("model-" + System.nanoTime());
        T5TestFixtures.writeConfig(modelDir, config);
        T5TestFixtures.writeSafeTensors(modelDir, T5TestFixtures.completeDenseT5Tensors(config));
        return modelDir;
    }
}
