package com.aresstack.windirectml.inference.decoderonly;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Device-free behaviour of the shared decoder-only generation loop, driven by a fake forward pass and selector. This
 * guards the control flow (token sequence, finish reason, streaming order, top-K diagnostics, profile label) that the
 * loop must keep identical across decoder-only families.
 */
class DecoderOnlyGenerationLoopTest {

    private static final int VOCAB = 8;

    @Test
    void stopsOnStopTokenAndReportsEosWithoutStreamingTheStopToken() {
        FakeForwardPass forwardPass = new FakeForwardPass(disabledProfile(),
                logitsWithArgmax(2), logitsWithArgmax(5), logitsWithArgmax(7));
        DecoderOnlyGenerationLoop loop = new DecoderOnlyGenerationLoop(forwardPass, 0, 0);
        List<Integer> streamed = new ArrayList<>();

        DecoderOnlyGenerationResult result = loop.generate(
                List.of(3, 1), 10, new GreedyStopSelector(7), streamed::add);

        assertEquals(List.of(2, 5, 7), result.generatedTokenIds());
        assertEquals(List.of(3, 1, 2, 5, 7), result.fullTokenIds());
        assertEquals(List.of(3, 1), result.inputTokenIds());
        assertEquals(3, result.tokensGenerated());
        assertEquals("eos_token", result.finishReason());
        assertEquals(10, result.maxNewTokens());
        // The stop token (7) must NOT be streamed; the accepted tokens before it are streamed in order.
        assertEquals(List.of(2, 5), streamed);
    }

    @Test
    void stopsAtMaxNewTokensWithLengthReason() {
        FakeForwardPass forwardPass = new FakeForwardPass(disabledProfile(),
                logitsWithArgmax(1), logitsWithArgmax(3));
        DecoderOnlyGenerationLoop loop = new DecoderOnlyGenerationLoop(forwardPass, 0, 0);
        List<Integer> streamed = new ArrayList<>();

        DecoderOnlyGenerationResult result = loop.generate(
                List.of(4), 2, new GreedyStopSelector(/*never*/ -1), streamed::add);

        assertEquals(List.of(1, 3), result.generatedTokenIds());
        assertEquals(2, result.tokensGenerated());
        assertEquals("length", result.finishReason());
        assertEquals(List.of(1, 3), streamed);
    }

    @Test
    void nullConsumerIsAllowed() {
        FakeForwardPass forwardPass = new FakeForwardPass(disabledProfile(),
                logitsWithArgmax(1), logitsWithArgmax(2));
        DecoderOnlyGenerationLoop loop = new DecoderOnlyGenerationLoop(forwardPass, 0, 0);

        DecoderOnlyGenerationResult result = loop.generate(List.of(0), 2, new GreedyStopSelector(-1), null);

        assertEquals(List.of(1, 2), result.generatedTokenIds());
    }

    @Test
    void recordsTopKDiagnosticsOnlyForTheFirstDebugSteps() {
        FakeForwardPass forwardPass = new FakeForwardPass(disabledProfile(),
                logitsWithArgmax(1), logitsWithArgmax(2), logitsWithArgmax(3));
        DecoderOnlyGenerationLoop loop = new DecoderOnlyGenerationLoop(forwardPass, 3, 2);

        DecoderOnlyGenerationResult result = loop.generate(List.of(0), 3, new GreedyStopSelector(-1), null);

        assertEquals(2, result.stepTopK().size());
        assertTrue(result.stepTopK().get(0).startsWith("warp top-3 @ step 0 (raw, pre-penalty): "),
                result.stepTopK().get(0));
        assertTrue(result.stepTopK().get(1).startsWith("warp top-3 @ step 1 (raw, pre-penalty): "));
    }

    @Test
    void enabledProfileEmitsMicroProfileWithInjectedLabel() {
        FakeForwardPass forwardPass = new FakeForwardPass(
                new DecoderOnlyWarpDecodeProfile(true, "SmolLM2"),
                logitsWithArgmax(1), logitsWithArgmax(2));
        DecoderOnlyGenerationLoop loop = new DecoderOnlyGenerationLoop(forwardPass, 0, 0);

        DecoderOnlyGenerationResult result = loop.generate(List.of(0), 2, new GreedyStopSelector(-1), null);

        assertFalse(result.decodeMicroProfile().isEmpty());
        assertTrue(result.decodeMicroProfile().get(0).startsWith("SmolLM2 WARP decode micro-profile (decode steps: "),
                result.decodeMicroProfile().get(0));
    }

    // ── test doubles ──────────────────────────────────────────────────────

    private static DecoderOnlyWarpDecodeProfile disabledProfile() {
        return new DecoderOnlyWarpDecodeProfile(false, "Test");
    }

    private static float[] logitsWithArgmax(int index) {
        float[] logits = new float[VOCAB];
        logits[index] = 1.0f;
        return logits;
    }

    /** Returns canned logits per step; ignores the KV cache (loop control-flow only). */
    private static final class FakeForwardPass implements DecoderOnlyForwardPass {
        private final DecoderOnlyConfig config =
                DecoderOnlyConfig.of(1, 4, 8, 1, 1, 2, 64, VOCAB, 1e-5f, 10000.0f);
        private final DecoderOnlyWarpDecodeProfile profile;
        private final List<float[]> logitsPerStep;
        private int call;

        FakeForwardPass(DecoderOnlyWarpDecodeProfile profile, float[]... logitsPerStep) {
            this.profile = profile;
            this.logitsPerStep = List.of(logitsPerStep);
        }

        @Override
        public DecoderOnlyConfig config() {
            return config;
        }

        @Override
        public DecoderOnlyWarpDecodeProfile decodeProfile() {
            return profile;
        }

        @Override
        public long lastCallLmHeadNanos() {
            return 0L;
        }

        @Override
        public float[] logitsForLastToken(List<Integer> tokenIds, DecoderOnlyWarpKvCache kvCache) {
            return logitsPerStep.get(call++);
        }
    }

    /** Greedy argmax selector that stops on a designated token id. */
    private static final class GreedyStopSelector implements DecoderOnlyTokenSelector {
        private final int stopId;

        GreedyStopSelector(int stopId) {
            this.stopId = stopId;
        }

        @Override
        public int selectNextToken(float[] logits, DecoderOnlyGeneratedTokens generatedTokens) {
            int best = 0;
            for (int i = 1; i < logits.length; i++) {
                if (logits[i] > logits[best]) {
                    best = i;
                }
            }
            return best;
        }

        @Override
        public boolean shouldStop(int tokenId) {
            return tokenId == stopId;
        }
    }
}
