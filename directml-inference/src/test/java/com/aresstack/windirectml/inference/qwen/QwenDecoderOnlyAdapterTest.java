package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyDecodeSession;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyForwardPass;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationLoop;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGenerationResult;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Slice 6 — experimental Qwen {@code DecoderOnlyDecodeSession} adapter. Device-free: a fake {@link QwenDecodeSteps}
 * stands in for Qwen's real (GPU-resident) decode pipeline, so the adapter wiring and the shared generation loop are
 * verified without a Qwen model or a WARP device. The production {@link Qwen2Runtime} path is not exercised here.
 */
class QwenDecoderOnlyAdapterTest {

    private static final int VOCAB = 8;

    private static Qwen2Config config() {
        return new Qwen2Config(896, 14, 24, 2, 151936, 32768, 4864, 1e-6f, 1_000_000f, true);
    }

    @BeforeEach
    void enableFlag() {
        System.setProperty(QwenDecoderOnlyForwardPass.EXPERIMENTAL_FLAG, "true");
    }

    @AfterEach
    void clearFlag() {
        System.clearProperty(QwenDecoderOnlyForwardPass.EXPERIMENTAL_FLAG);
    }

    @Test
    void factoryRefusesConstructionWhenFlagDisabled() {
        System.clearProperty(QwenDecoderOnlyForwardPass.EXPERIMENTAL_FLAG);
        FakeSteps steps = new FakeSteps(logitsWithArgmax(1));
        assertThrows(IllegalStateException.class, () -> new QwenDecoderOnlyForwardPass(config(), steps));
    }

    @Test
    void runtimePropertyAloneEnablesTheSessionPath() {
        // The real opt-in -Dqwen.runtime=decoderonly-session enables the path without the experimental flag.
        System.clearProperty(QwenDecoderOnlyForwardPass.EXPERIMENTAL_FLAG);
        System.setProperty("qwen.runtime", "decoderonly-session");
        try {
            assertTrue(QwenDecoderOnlyForwardPass.experimentalEnabled());
            DecoderOnlyForwardPass forwardPass =
                    new QwenDecoderOnlyForwardPass(config(), new FakeSteps(logitsWithArgmax(1)));
            assertNotNull(forwardPass.newDecodeSession(8));
        } finally {
            System.clearProperty("qwen.runtime");
        }
    }

    @Test
    void sessionResetsThenPrefillsThenDecodesThroughQwenSteps() {
        FakeSteps steps = new FakeSteps(logitsWithArgmax(2), logitsWithArgmax(5));
        DecoderOnlyForwardPass forwardPass = new QwenDecoderOnlyForwardPass(config(), steps);

        try (DecoderOnlyDecodeSession session = forwardPass.newDecodeSession(64)) {
            float[] prefillLogits = session.prefill(List.of(9, 9, 9));
            assertEquals(2, argmax(prefillLogits));
            assertEquals(1, steps.resets);                 // reset happens before prefill
            assertEquals(List.of(3), steps.prefillTokenCounts); // whole 3-token prompt prefilled at once

            float[] decodeLogits = session.decodeNext(2);
            assertEquals(5, argmax(decodeLogits));
            assertEquals(List.of(2), steps.decoded);       // exactly the previously selected token
        }
        // close() releases this run's decode state from the shared runtime.
        assertEquals(2, steps.resets);
    }

    @Test
    void decodeNextBeforePrefillIsRejected() {
        FakeSteps steps = new FakeSteps(logitsWithArgmax(1));
        DecoderOnlyForwardPass forwardPass = new QwenDecoderOnlyForwardPass(config(), steps);
        DecoderOnlyDecodeSession session = forwardPass.newDecodeSession(64);
        assertThrows(IllegalStateException.class, () -> session.decodeNext(1));
    }

    @Test
    void qwenSessionDrivesTheSharedGenerationLoopWithQwenTokenSelector() {
        FakeSteps steps = new FakeSteps(logitsWithArgmax(1), logitsWithArgmax(3));
        DecoderOnlyForwardPass forwardPass = new QwenDecoderOnlyForwardPass(config(), steps);
        DecoderOnlyGenerationLoop loop = new DecoderOnlyGenerationLoop(forwardPass, 0, 0);
        List<Integer> streamed = new ArrayList<>();

        DecoderOnlyGenerationResult result = loop.generate(
                List.of(7, 8), 2, QwenTokenSelector.greedy(1.0f), streamed::add);

        assertEquals(List.of(1, 3), result.generatedTokenIds());
        assertEquals(List.of(7, 8, 1, 3), result.fullTokenIds());
        assertEquals("length", result.finishReason()); // small argmax ids are never Qwen stop tokens
        assertEquals(List.of(1, 3), streamed);
        // Loop drove the Qwen session: prefill of the 2-token prompt, then one decodeNext of the first selected token.
        assertEquals(List.of(2), steps.prefillTokenCounts);
        assertEquals(List.of(1), steps.decoded);
    }

    // ── test doubles ──────────────────────────────────────────────────────

    private static float[] logitsWithArgmax(int index) {
        float[] logits = new float[VOCAB];
        logits[index] = 1.0f;
        return logits;
    }

    private static int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    /** Stands in for Qwen's real decode pipeline; records calls and returns canned logits per step. */
    private static final class FakeSteps implements QwenDecodeSteps {
        private final List<float[]> logitsPerStep;
        private final List<Integer> prefillTokenCounts = new ArrayList<>();
        private final List<Integer> decoded = new ArrayList<>();
        private int resets;
        private int call;

        FakeSteps(float[]... logitsPerStep) {
            this.logitsPerStep = List.of(logitsPerStep);
        }

        @Override
        public void resetDecodeState() {
            resets++;
        }

        @Override
        public float[] decodePrefill(int[] promptTokenIds) {
            prefillTokenCounts.add(promptTokenIds.length);
            return logitsPerStep.get(call++);
        }

        @Override
        public float[] decodeNextToken(int tokenId) {
            decoded.add(tokenId);
            return logitsPerStep.get(call++);
        }
    }
}
