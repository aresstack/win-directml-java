package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GEMMA-WARP-15: the decode dispatch breakdown is validated against the empirically measured per-token
 * dispatch counts (380 before the norm+add fusion — the 14b measurement — and 344 after), and the largest
 * fuseable small-WARP-dispatch group it identifies is the RMSNorm/residual-add chain that GEMMA-WARP-15
 * fuses. Pure CPU model, runs everywhere.
 */
class Gemma3DecodeDispatchBreakdownTest {

    private static final int GEMMA_LAYERS = 18; // real Gemma 3 270M

    @Test
    void baselineMatchesTheMeasured380DispatchesPerToken() {
        Gemma3DecodeDispatchBreakdown b = Gemma3DecodeDispatchBreakdown.baseline(GEMMA_LAYERS);
        assertEquals(21, b.dispatchesPerLayer(), "21 dispatches/layer pre-fusion");
        assertEquals(380, b.dispatchesPerToken(), "matches the GEMMA-WARP-14b measurement");
    }

    @Test
    void currentBreakdownReflectsNormAddAndProjectionFusion() {
        Gemma3DecodeDispatchBreakdown c = Gemma3DecodeDispatchBreakdown.current(GEMMA_LAYERS);
        // 21 (pre-15) - 2 (15 norm+add) - 3 (16: qkv 3->1, gate/up 2->1) = 16 dispatches/layer.
        assertEquals(16, c.dispatchesPerLayer(), "16 dispatches/layer after norm+add + projection fusion");
        assertEquals(290, c.dispatchesPerToken(), "16/layer * 18 + 2 tail");
    }

    @Test
    void projectionFusionRemovesThreeDispatchesPerLayer() {
        // GEMMA-WARP-16 alone: qkv (3->1) + gate/up (2->1) = 3 fewer dispatches/layer vs the 15 state (19/layer).
        int after16 = Gemma3DecodeDispatchBreakdown.current(GEMMA_LAYERS).dispatchesPerToken();
        int state15 = 19 * GEMMA_LAYERS + 2; // 344
        assertEquals(3 * GEMMA_LAYERS, state15 - after16, "projection fusion removes 3 dispatches/layer");
    }

    @Test
    void allFusionsTogetherRemove90DispatchesPerToken() {
        int before = Gemma3DecodeDispatchBreakdown.baseline(GEMMA_LAYERS).dispatchesPerToken();
        int after = Gemma3DecodeDispatchBreakdown.current(GEMMA_LAYERS).dispatchesPerToken();
        assertEquals(90, before - after, "380 -> 290 across 15 (norm+add) + 16 (projection) fusions");
    }

    @Test
    void largestSmallDispatchGroupsAreTheFusionTargets() {
        List<Gemma3DecodeDispatchBreakdown.Group> small =
                Gemma3DecodeDispatchBreakdown.baseline(GEMMA_LAYERS).smallDispatchGroupsLargestFirst();
        assertFalse(small.isEmpty());
        // The top groups are the 2/layer small-dispatch chains (rmsnorm, residual-add, qk-norm, rope, kv).
        assertEquals(2, small.get(0).perLayer(), "largest small groups are 2/layer");
        // The DML-GEMM projections must never appear as fusion targets.
        assertTrue(small.stream().noneMatch(Gemma3DecodeDispatchBreakdown.Group::dmlGemm),
                "DML-GEMM projections are not fusion targets");
        // The post-attn/post-ff rmsnorm + the residual-add (the GEMMA-WARP-15 fusion) are among the largest.
        assertTrue(small.stream().anyMatch(g -> g.name().contains("rmsnorm post-attn")));
        assertTrue(small.stream().anyMatch(g -> g.name().contains("residual element-add")));
    }
}
