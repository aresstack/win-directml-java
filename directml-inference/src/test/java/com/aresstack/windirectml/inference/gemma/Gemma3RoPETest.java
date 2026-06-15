package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GEMMA-WARP-7a (device-free): the RoPE reference across heads and the local/global attention layout
 * (complements {@link Gemma3AttentionLayoutTest}: all-layer full/local classification, explicit
 * head_dim=256, and GQA mapping with more than one kv head).
 */
class Gemma3RoPETest {

    private static Gemma3Config config(int numHeads, int numKvHeads, int headDim, int window) {
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                640, 2048, 18, numHeads, numKvHeads, headDim, 262144, 32768, 1e-6,
                1_000_000, 10_000, window, 6, List.of(), 256, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    // ── RoPE reference ───────────────────────────────────────────────

    @Test
    void ropeAcrossHeadsMatchesPerHeadReference() {
        int numHeads = 3;
        int headDim = 8;
        int pos = 5;
        double theta = 10_000.0;
        Random rng = new Random(101);
        float[] packed = new float[numHeads * headDim];
        for (int i = 0; i < packed.length; i++) {
            packed[i] = rng.nextFloat() * 2 - 1;
        }
        float[] expected = packed.clone();
        for (int head = 0; head < numHeads; head++) {
            Gemma3ReferenceMath.applyRopeHalf(expected, head * headDim, headDim, pos, theta);
        }
        float[] got = packed.clone();
        Gemma3RoPE.applyToHeads(got, numHeads, headDim, pos, theta);
        assertEquals(expected.length, got.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], got[i], 1e-6f, "rope[" + i + "]");
        }
    }

    @Test
    void ropeAtPositionZeroIsIdentity() {
        Random rng = new Random(103);
        float[] packed = new float[2 * 8];
        for (int i = 0; i < packed.length; i++) {
            packed[i] = rng.nextFloat() * 2 - 1;
        }
        float[] original = packed.clone();
        Gemma3RoPE.applyToHeads(packed, 2, 8, 0, 10_000.0);
        for (int i = 0; i < packed.length; i++) {
            assertEquals(original[i], packed[i], 1e-6f, "pos0[" + i + "]");
        }
    }

    @Test
    void ropePreservesEachPairNorm() {
        // Rotation is orthogonal: |(x_i, x_{i+half})| is unchanged.
        int headDim = 256; // real Gemma head_dim
        float[] packed = new float[headDim];
        Random rng = new Random(107);
        for (int i = 0; i < headDim; i++) {
            packed[i] = rng.nextFloat() * 2 - 1;
        }
        float[] before = packed.clone();
        Gemma3RoPE.applyToHeads(packed, 1, headDim, 9, 1_000_000.0);
        int half = headDim / 2;
        for (int i = 0; i < half; i++) {
            double nBefore = Math.hypot(before[i], before[i + half]);
            double nAfter = Math.hypot(packed[i], packed[i + half]);
            assertEquals(nBefore, nAfter, 1e-4, "pair norm[" + i + "]");
        }
    }

    // ── layout: full/local across all layers, head_dim, GQA ──────────

    @Test
    void fullAndLocalLayersAcrossAllLayers() {
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(4, 1, 256, 512));
        for (int layer = 0; layer < 18; layer++) {
            boolean expectFull = (layer == 5 || layer == 11 || layer == 17); // pattern 6
            assertEquals(expectFull, layout.isFullAttention(layer), "layer " + layer + " full?");
            assertEquals(expectFull ? 1_000_000.0 : 10_000.0, layout.ropeTheta(layer), 1e-6,
                    "layer " + layer + " theta");
        }
    }

    @Test
    void headDimIsExplicitNotHiddenOverHeads() {
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(4, 1, 256, 512));
        assertEquals(256, layout.headDim());
        assertFalse(640 / 4 == layout.headDim(), "head_dim must be the config value, not hidden/heads");
    }

    @Test
    void gqaMappingWithMultipleKvHeads() {
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(8, 2, 256, 512));
        assertEquals(4, layout.groupsPerKvHead()); // 8 heads / 2 kv
        for (int h = 0; h < 4; h++) {
            assertEquals(0, layout.kvHeadFor(h), "head " + h);
        }
        for (int h = 4; h < 8; h++) {
            assertEquals(1, layout.kvHeadFor(h), "head " + h);
        }
    }

    @Test
    void slidingWindowAndCausalAcrossLayers() {
        int window = 512;
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(4, 1, 256, window));
        // local layer: bounded window
        assertEquals(Math.max(0, 1000 - window + 1), layout.firstValidKey(0, 1000));
        assertTrue(layout.isVisible(0, 1000, 1000 - window + 1));
        assertFalse(layout.isVisible(0, 1000, 1000 - window));   // outside window
        assertFalse(layout.isVisible(0, 1000, 1001));            // future
        // full layer: from 0, still causal
        assertEquals(0, layout.firstValidKey(5, 1000));
        assertTrue(layout.isVisible(5, 1000, 0));
        assertFalse(layout.isVisible(5, 1000, 1001));
    }
}
