package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-7b: GPU correctness of the Gemma WARP RoPE and attention-scores kernels against the
 * verified CPU reference ({@link Gemma3ReferenceMath#applyRopeHalf} / {@link Gemma3RoPE} and the scaled
 * masked {@code QK^T} of {@code Gemma3ReferenceForwardPass}, with the visible range from
 * {@link Gemma3AttentionLayout}).
 *
 * <p>Skipped (assumption-aborted, not failed) without a DirectML/D3D12 device, so it stays green on CI.
 * The GPU computes in {@code float}.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpAttentionKernelTest {

    private static final float ABS_TOL = 1e-4f;
    private static final float REL_TOL = 1e-4f;

    private static final int GEMMA_HEAD_DIM = 256;

    private static WindowsBindings wb;

    @BeforeAll
    static void initGpu() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        wb = new WindowsBindings();
        wb.init("directml");
    }

    @AfterAll
    static void closeGpu() {
        if (wb != null) {
            wb.close();
        }
    }

    private static Gemma3Config config(int window) {
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                640, 2048, 18, 4, 1, GEMMA_HEAD_DIM, 262144, 32768, 1e-6,
                1_000_000, 10_000, window, 6, List.of(), 256, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    // ── RoPE kernel ──────────────────────────────────────────────────

    @Test
    void warpRoPEMatchesReferenceOnSmallShape() throws Exception {
        assertRoPECloseToReference(2, 8, 3, 10_000f, new Random(201));
    }

    @Test
    void warpRoPEMatchesReferenceOnRealHeadDimBothThetas() throws Exception {
        // Real Gemma: head_dim=256, local theta 1e4 and global theta 1e6, several positions.
        for (float theta : new float[]{10_000f, 1_000_000f}) {
            for (int pos : new int[]{0, 1, 7, 31}) {
                assertRoPECloseToReference(4, GEMMA_HEAD_DIM, pos, theta, new Random(202 + pos));
            }
        }
    }

    @Test
    void warpRoPEAtPositionZeroIsIdentity() throws Exception {
        Random rng = new Random(211);
        float[] packed = randomTensor(rng, 4 * GEMMA_HEAD_DIM);
        try (Gemma3WarpRoPEKernel kernel = new Gemma3WarpRoPEKernel(wb)) {
            float[] gpu = kernel.applyToHeads(packed, 4, GEMMA_HEAD_DIM, 0, 1_000_000f);
            for (int i = 0; i < packed.length; i++) {
                assertClose("rope0[" + i + "]", packed[i], gpu[i]);
            }
        }
    }

    // ── attention scores kernel ──────────────────────────────────────

    @Test
    void warpScoresFullLayerMatchesReference() throws Exception {
        // Full-attention layer (5): firstValid = 0, GQA 4 heads / 1 kv, head_dim=256.
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(512));
        int layer = 5;
        int seqLen = 12;
        int queryPos = 9;
        assertScoresCloseToReference(layout, layer, 4, 1, GEMMA_HEAD_DIM, seqLen, queryPos, new Random(221));
    }

    @Test
    void warpScoresLocalLayerHonoursSlidingWindow() throws Exception {
        // Local layer (0) with a small window so firstValid > 0 and the mask actually bites.
        int window = 4;
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(window));
        int layer = 0;
        int seqLen = 16;
        int queryPos = 10; // firstValid = 10 - 4 + 1 = 7
        assertScoresCloseToReference(layout, layer, 4, 1, GEMMA_HEAD_DIM, seqLen, queryPos, new Random(223));
    }

    @Test
    void warpScoresGqaMapsHeadsToKvHeads() throws Exception {
        // 4 heads / 2 kv -> groupsPerKv 2; head_dim small to keep the test light.
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(
                new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                        640, 2048, 18, 4, 2, 8, 262144, 32768, 1e-6,
                        1_000_000, 10_000, 512, 6, List.of(), 256, "gelu_pytorch_tanh", 2, 1, 0, true));
        assertScoresCloseToReference(layout, 5, 4, 2, 8, 6, 4, new Random(227));
    }

    // ── helpers ──────────────────────────────────────────────────────

    private void assertRoPECloseToReference(int numHeads, int headDim, int pos, float theta, Random rng)
            throws Exception {
        float[] packed = randomTensor(rng, numHeads * headDim);
        float[] expected = packed.clone();
        Gemma3RoPE.applyToHeads(expected, numHeads, headDim, pos, theta);
        try (Gemma3WarpRoPEKernel kernel = new Gemma3WarpRoPEKernel(wb)) {
            float[] gpu = kernel.applyToHeads(packed, numHeads, headDim, pos, theta);
            for (int i = 0; i < expected.length; i++) {
                assertClose("rope[" + i + "] (pos=" + pos + ", theta=" + theta + ", headDim=" + headDim + ")",
                        expected[i], gpu[i]);
            }
        }
    }

    private void assertScoresCloseToReference(Gemma3AttentionLayout layout, int layer, int numHeads,
                                              int numKvHeads, int headDim, int seqLen, int queryPos,
                                              Random rng) throws Exception {
        int kvDim = numKvHeads * headDim;
        float[] q = randomTensor(rng, numHeads * headDim);
        float[] keys = randomTensor(rng, seqLen * kvDim);
        int firstValid = layout.firstValidKey(layer, queryPos);
        float scale = layout.attentionScale();

        float[] expected = referenceScores(q, keys, numHeads, numKvHeads, headDim, seqLen, queryPos,
                firstValid, scale);

        try (Gemma3WarpAttentionScoresKernel kernel = new Gemma3WarpAttentionScoresKernel(wb)) {
            float[] gpu = kernel.scores(q, keys, numHeads, numKvHeads, headDim, seqLen, queryPos,
                    firstValid, scale);
            assertEquals(expected.length, gpu.length, "scores length");
            for (int head = 0; head < numHeads; head++) {
                for (int j = 0; j < seqLen; j++) {
                    int idx = head * seqLen + j;
                    if (expected[idx] == Gemma3WarpAttention.SCORE_SENTINEL) {
                        assertEquals(Gemma3WarpAttention.SCORE_SENTINEL, gpu[idx], 0f,
                                "masked score head=" + head + " j=" + j);
                    } else {
                        assertClose("score head=" + head + " j=" + j, expected[idx], gpu[idx]);
                    }
                }
            }
        }
    }

    private static float[] referenceScores(float[] q, float[] keys, int numHeads, int numKvHeads,
                                           int headDim, int seqLen, int queryPos, int firstValid,
                                           float scale) {
        int kvDim = numKvHeads * headDim;
        int groupsPerKv = numHeads / numKvHeads;
        float[] out = new float[numHeads * seqLen];
        for (int head = 0; head < numHeads; head++) {
            int kvHead = head / groupsPerKv;
            for (int j = 0; j < seqLen; j++) {
                if (j < firstValid || j > queryPos) {
                    out[head * seqLen + j] = Gemma3WarpAttention.SCORE_SENTINEL;
                    continue;
                }
                float acc = 0;
                int qBase = head * headDim;
                int kBase = j * kvDim + kvHead * headDim;
                for (int c = 0; c < headDim; c++) {
                    acc += q[qBase + c] * keys[kBase + c];
                }
                out[head * seqLen + j] = acc * scale;
            }
        }
        return out;
    }

    private static float[] randomTensor(Random rng, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rng.nextFloat() * 2 - 1;
        }
        return v;
    }

    private static void assertClose(String label, float want, float got) {
        float tol = ABS_TOL + REL_TOL * Math.abs(want);
        float diff = Math.abs(want - got);
        if (diff > tol) {
            fail(String.format("%s mismatch: want=%.6f got=%.6f diff=%.3e (tol=%.3e)",
                    label, want, got, diff, tol));
        }
        assertEquals(want, got, tol, label);
    }
}
