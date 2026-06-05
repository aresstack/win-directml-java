package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2WarpRuntimePlannerTest {

    @TempDir
    Path tempDir;

    @Test
    void plansWeightsKvCacheScratchAndKernelSkeleton() throws Exception {
        SmolLM2Weights weights = loadWeights(SmolLM2TestFixtures.config135(false), true);
        SmolLM2WarpRuntimePlan plan = new SmolLM2WarpRuntimePlanner().plan(weights, 16);

        assertTrue(plan.readyForNativeAllocation());
        assertEquals(16, plan.sequenceLength());
        assertEquals(0, plan.bufferPlan().totalBytes() % SmolLM2WarpRuntimePlanner.DEFAULT_ALIGNMENT_BYTES);
        assertFalse(plan.bufferPlan().entriesOfKind(SmolLM2WarpBufferKind.WEIGHT).isEmpty());
        assertEquals(2, plan.bufferPlan().entriesOfKind(SmolLM2WarpBufferKind.KV_CACHE).size());
        assertEquals(8, plan.bufferPlan().entriesOfKind(SmolLM2WarpBufferKind.SCRATCH).size());
        assertEquals(10, plan.kernelPlan().stepCount());
    }

    @Test
    void tiedEmbeddingsAreNotUploadedTwiceAsLmHead() throws Exception {
        SmolLM2Weights weights = loadWeights(SmolLM2TestFixtures.config135(true), false);
        SmolLM2WarpRuntimePlan plan = new SmolLM2WarpRuntimePlanner().plan(weights, 8);

        long lmHeadUploads = plan.bufferPlan().entriesOfKind(SmolLM2WarpBufferKind.WEIGHT).stream()
                .filter(entry -> entry.role() == SmolLM2TensorRole.LM_HEAD)
                .count();

        assertEquals(0L, lmHeadUploads);
    }

    @Test
    void reportsUnsupportedBiasBuffersAsWarnings() throws Exception {
        SmolLM2Weights supportedWeights = loadWeights(SmolLM2TestFixtures.config135(false), true);
        SmolLM2Config warningConfig = new SmolLM2Config("llama", java.util.List.of("LlamaForCausalLM"),
                8, 16, 1, 4, 2, 2, 32, 64,
                1.0e-5d, 10000.0d, "gelu", true, true, 1, 2, null, false);
        SmolLM2Weights weights = new SmolLM2Weights(
                warningConfig,
                supportedWeights.tokenEmbedding(),
                supportedWeights.finalNorm(),
                supportedWeights.lmHead(),
                supportedWeights.lmHeadTiedToEmbedding(),
                supportedWeights.layers(),
                supportedWeights.payloadBytes());

        SmolLM2WarpRuntimePlan plan = new SmolLM2WarpRuntimePlanner().plan(weights, 8);

        assertFalse(plan.readyForNativeAllocation());
        assertEquals(3, plan.warnings().size());
    }

    private SmolLM2Weights loadWeights(SmolLM2Config config, boolean includeLmHead) throws Exception {
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, includeLmHead);
        Path output = tempDir.resolve("model-" + System.nanoTime() + ".wdmlpack");
        new SmolLM2WdmlPackCompiler().compile(new SmolLM2CompileOptions(tempDir, output, false, false));
        return SmolLM2RuntimePackage.open(output).requireWeights();
    }
}
