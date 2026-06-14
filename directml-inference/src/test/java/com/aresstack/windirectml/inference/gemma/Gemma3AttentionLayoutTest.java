package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** GEMMA-WARP-7 (device-free): attention layout, GQA mapping, causal + sliding-window visibility. */
class Gemma3AttentionLayoutTest {

    private static Gemma3Config config(int window) {
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                640, 2048, 18, 4, 1, 256, 262144, 32768, 1e-6, 1_000_000, 10_000, window, 6,
                List.of(), 256, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    @Test
    void gqaMapsAllQueryHeadsToTheSingleKvHead() {
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(512));
        assertEquals(4, layout.groupsPerKvHead()); // 4 heads / 1 kv
        for (int h = 0; h < 4; h++) {
            assertEquals(0, layout.kvHeadFor(h));
        }
    }

    @Test
    void fullAttentionLayersAttendFromZero() {
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(4));
        int fullLayer = 5; // pattern 6 -> full at 5,11,17
        assertTrue(layout.isFullAttention(fullLayer));
        assertEquals(0, layout.firstValidKey(fullLayer, 100));
        assertTrue(layout.isVisible(fullLayer, 100, 0));   // global sees the start
        assertFalse(layout.isVisible(fullLayer, 5, 6));    // still causal
    }

    @Test
    void slidingLayersHonourTheWindow() {
        int window = 4;
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(window));
        int local = 0; // sliding
        assertFalse(layout.isFullAttention(local));
        assertEquals(7, layout.firstValidKey(local, 10)); // 10-4+1
        assertTrue(layout.isVisible(local, 10, 10));       // self
        assertTrue(layout.isVisible(local, 10, 7));        // window edge
        assertFalse(layout.isVisible(local, 10, 6));       // outside window
        assertFalse(layout.isVisible(local, 10, 11));      // future (causal)
    }

    @Test
    void dualThetaPerLayer() {
        Gemma3AttentionLayout layout = new Gemma3AttentionLayout(config(512));
        assertEquals(1_000_000.0, layout.ropeTheta(5), 1e-6);
        assertEquals(10_000.0, layout.ropeTheta(0), 1e-6);
        assertEquals(1.0f / (float) Math.sqrt(256), layout.attentionScale(), 1e-7);
    }
}
