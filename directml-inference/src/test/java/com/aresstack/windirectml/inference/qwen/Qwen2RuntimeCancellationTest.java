package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link Qwen2Runtime} cancellation and progress listener API.
 *
 * <p>Cancellation-flag tests use a minimal {@link Qwen2Config} (tiny dimensions)
 * with {@code null} weights and tokenizer, which is sufficient because those
 * fields are not accessed during construction or by the cancel/isCancelled/
 * resetCancelled methods.
 */
class Qwen2RuntimeCancellationTest {

    /** Minimal config: 14 Q heads / 2 KV heads → valid Grouped Query Attention (GQA) ratio of 7:1, tiny dimensions. */
    private static Qwen2Config minimalConfig() {
        return new Qwen2Config(
                /*hiddenSize*/              14,
                /*numAttentionHeads*/       14,
                /*numHiddenLayers*/         2,
                /*numKeyValueHeads*/        2,
                /*vocabSize*/               100,
                /*maxPositionEmbeddings*/   128,
                /*intermediateSize*/        56,
                /*rmsNormEps*/              1e-6f,
                /*ropeTheta*/               1_000_000f,
                /*tieWordEmbeddings*/       false
        );
    }

    @Test
    void cancelFlagIsInitiallyFalse() {
        Qwen2Runtime runtime = new Qwen2Runtime(minimalConfig(), null, null);
        assertFalse(runtime.isCancelled(), "cancel flag must be false on construction");
    }

    @Test
    void cancelMethodSetsCancelledFlag() {
        Qwen2Runtime runtime = new Qwen2Runtime(minimalConfig(), null, null);
        assertFalse(runtime.isCancelled());
        runtime.cancel();
        assertTrue(runtime.isCancelled(), "cancel() must set isCancelled() to true");
    }

    @Test
    void resetCancelledClearsFlagAfterCancel() {
        Qwen2Runtime runtime = new Qwen2Runtime(minimalConfig(), null, null);
        runtime.cancel();
        assertTrue(runtime.isCancelled());
        runtime.resetCancelled();
        assertFalse(runtime.isCancelled(), "resetCancelled() must clear the cancelled flag");
    }

    @Test
    void prefillProgressListenerCanBeConstructedFromLambda() {
        Qwen2Runtime.PrefillProgressListener listener = (layer, totalLayers, elapsedMs, seqLen) -> {};
        assertNotNull(listener);
    }

    @Test
    void prefillProgressListenerReceivesCorrectParameters() {
        List<String> events = new ArrayList<>();
        Qwen2Runtime.PrefillProgressListener listener = (layer, totalLayers, elapsedMs, seqLen) -> {
            events.add(layer + "/" + totalLayers + " " + elapsedMs + "ms seq=" + seqLen);
        };

        // Values approximate the real-world measurements (~200 s per layer on Java CPU path)
        // to make the test representative of the actual performance profile being guarded.
        listener.onLayerComplete(1, 24, 202020, 35);
        listener.onLayerComplete(4, 24, 198000, 35);
        listener.onLayerComplete(24, 24, 205000, 35);

        assertEquals(3, events.size());
        assertEquals("1/24 202020ms seq=35", events.get(0));
        assertEquals("4/24 198000ms seq=35", events.get(1));
        assertEquals("24/24 205000ms seq=35", events.get(2));
    }

    @Test
    void tokenConsumerInterfaceAcceptsDelta() {
        List<String> deltas = new ArrayList<>();
        Qwen2Runtime.TokenConsumer consumer = (tokenId, full, delta) -> deltas.add(delta);

        consumer.onToken(42, "Hello", "Hello");
        consumer.onToken(43, "Hello world", " world");

        assertEquals(2, deltas.size());
        assertEquals("Hello", deltas.get(0));
        assertEquals(" world", deltas.get(1));
    }
}
