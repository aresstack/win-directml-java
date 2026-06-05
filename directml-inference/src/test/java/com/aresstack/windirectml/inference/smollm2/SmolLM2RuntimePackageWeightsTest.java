package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2RuntimePackageWeightsTest {

    @TempDir
    Path tempDir;

    @Test
    void compilerWritesPayloadAndRuntimePackageResolvesWeights() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, true);
        Path output = tempDir.resolve("smol-runtime.wdmlpack");

        SmolLM2CompileReport report = new SmolLM2WdmlPackCompiler().compile(
                new SmolLM2CompileOptions(tempDir, output, false, false));
        SmolLM2RuntimePackage runtimePackage = SmolLM2RuntimePackage.open(output);
        SmolLM2Weights weights = runtimePackage.requireWeights();

        assertTrue(report.payloadIncluded());
        assertTrue(RuntimeModelPackage.open(output).payloadIncluded());
        assertTrue(runtimePackage.executable());
        assertTrue(runtimePackage.weightsAvailable());
        assertEquals(config.hiddenSize(), runtimePackage.config().hiddenSize());
        assertEquals(config.numHiddenLayers(), weights.layers().size());
        assertFalse(weights.lmHeadTiedToEmbedding());
        assertTrue(weights.tokenEmbedding().rawByteLength() > 0);
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                weights.layer(0).require(SmolLM2TensorRole.LAYER_SELF_Q).tensorName());
    }

    @Test
    void tiedEmbeddingsUseTokenEmbeddingAsLmHead() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(true);
        SmolLM2TestFixtures.writeModelDirectory(tempDir, config, false);
        Path output = tempDir.resolve("smol-tied.wdmlpack");

        new SmolLM2WdmlPackCompiler().compile(new SmolLM2CompileOptions(tempDir, output, false, false));
        SmolLM2Weights weights = SmolLM2RuntimePackage.open(output).requireWeights();

        assertTrue(weights.lmHeadTiedToEmbedding());
        assertEquals(weights.tokenEmbedding().tensorName(), weights.lmHead().tensorName());
    }

    @Test
    void manifestOnlyPackageDoesNotExposeWeights() throws Exception {
        SmolLM2Config config = SmolLM2TestFixtures.config135(false);
        SmolLM2LayoutReport layoutReport = new SmolLM2LayoutValidator().validate(
                config, SmolLM2TestFixtures.completeCatalog(config, true));
        Path output = tempDir.resolve("manifest-only.wdmlpack");

        new SmolLM2WdmlPackManifestWriter().writeManifestOnly(output, config, layoutReport);
        SmolLM2RuntimePackage runtimePackage = SmolLM2RuntimePackage.open(output);

        assertFalse(runtimePackage.weightsAvailable());
        assertThrows(IllegalStateException.class, runtimePackage::requireWeights);
    }
}
