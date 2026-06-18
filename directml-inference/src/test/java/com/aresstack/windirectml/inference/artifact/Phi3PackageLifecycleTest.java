package com.aresstack.windirectml.inference.artifact;

import com.aresstack.windirectml.inference.phi3.Phi3CompileOptions;
import com.aresstack.windirectml.inference.phi3.Phi3Config;
import com.aresstack.windirectml.inference.phi3.Phi3WdmlPackCompiler;
import com.aresstack.windirectml.inference.phi3.Phi3Weights;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.windirectml.inference.phi3.Phi3Weights.QuantizedWeight;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * PHI3-WORKBENCH-RUNNABLE-1: the Phi-3 lifecycle is now compiler-backed and detects a compiled package as a valid
 * runtime package. These are light tests (a tiny synthetic package) -- they do not reconstruct the full ~3 GB weights.
 */
class Phi3PackageLifecycleTest {

    @TempDir
    Path tempDir;

    @Test
    void isCompilerBacked() {
        Phi3PackageLifecycle lifecycle = new Phi3PackageLifecycle();
        assertEquals(ModelFamily.PHI3, lifecycle.family());
        assertTrue(lifecycle.hasCompiler(), "Phi-3 is now compiler-backed (model_phi3.wdmlpack)");
    }

    @Test
    void missingPackageIsNotReady() {
        ModelArtifactStatus status = new Phi3PackageLifecycle().inspect(tempDir);
        assertEquals(PackageState.PACKAGE_MISSING, status.packageState());
        assertFalse(status.executable());
        assertFalse(status.ready());
    }

    @Test
    void compiledPackageIsValidAndExecutable() throws Exception {
        // Write a tiny synthetic model_phi3.wdmlpack via the in-memory compiler core (no ONNX, no 3 GB weights).
        Phi3Config config = new Phi3Config(4, 2, 1, 2, 6, 8, 8, 1e-5f, 10000.0f);
        Phi3WdmlPackCompiler.writePackage(config, syntheticWeights(config),
                tempDir.resolve(Phi3CompileOptions.DEFAULT_OUTPUT_NAME), true);

        ModelArtifactStatus status = new Phi3PackageLifecycle().inspect(tempDir);
        assertEquals(PackageState.PACKAGE_VALID, status.packageState());
        assertTrue(status.executable());
        assertTrue(status.ready(), "a compiled Phi-3 wdmlpack must be detected as ready");
    }

    private static Phi3Weights syntheticWeights(Phi3Config c) {
        int h = c.hiddenSize();
        int inter = c.intermediateSize();
        LayerWeights layer = new LayerWeights(
                ramp(h), quant(h, h, 4), quant(h, h, 4), quant(h, h, 4), ramp(h), quant(h, h, 4),
                ramp(h), quant(2 * inter, h, 4), ramp(inter), quant(h, inter, 4));
        return Phi3Weights.ofRecords(c, ramp(c.vocabSize() * h), ramp(8), ramp(8),
                new LayerWeights[]{layer}, ramp(h), quant(c.vocabSize(), h, 4));
    }

    private static float[] ramp(int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = i * 0.1f;
        }
        return out;
    }

    private static QuantizedWeight quant(int n, int k, int blockSize) {
        int blocksPerRow = k / blockSize;
        byte[] qWeight = new byte[n * blocksPerRow * (blockSize / 2)];
        float[] scales = new float[n * blocksPerRow];
        byte[] zp = new byte[(n * blocksPerRow + 1) / 2];
        return new QuantizedWeight(qWeight, scales, zp, n, k, blockSize);
    }
}
