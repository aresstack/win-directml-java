package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** GEMMA-WARP-10 (reference): greedy generation loop runs, stops at maxTokens/EOS, is deterministic. */
class Gemma3ReferenceGeneratorTest {

    private static Gemma3Config tinyConfig() {
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                8, 16, 2, 2, 1, 4, 20, 64, 1e-6, 1_000_000, 10_000, 3, 0,
                List.of(Gemma3Config.SLIDING_ATTENTION, Gemma3Config.FULL_ATTENTION),
                4, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    private static Gemma3ReferenceWeights randomWeights(Gemma3Config c) {
        Random rng = new Random(7);
        int h = c.hiddenSize();
        Gemma3ReferenceWeights.Layer[] layers = new Gemma3ReferenceWeights.Layer[c.numHiddenLayers()];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Gemma3ReferenceWeights.Layer(
                    rand(rng, h), rand(rng, c.attentionDim() * h), rand(rng, c.keyValueDim() * h),
                    rand(rng, c.keyValueDim() * h), rand(rng, h * c.attentionDim()),
                    rand(rng, c.headDim()), rand(rng, c.headDim()), rand(rng, h), rand(rng, h),
                    rand(rng, c.intermediateSize() * h), rand(rng, c.intermediateSize() * h),
                    rand(rng, h * c.intermediateSize()), rand(rng, h));
        }
        return new Gemma3ReferenceWeights(c, rand(rng, c.vocabSize() * h), rand(rng, h), layers);
    }

    private static float[] rand(Random rng, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (rng.nextFloat() - 0.5f) * 0.1f;
        }
        return a;
    }

    @Test
    void generatesUpToMaxTokensDeterministicallyWithCallback() {
        Gemma3Config c = tinyConfig();
        Gemma3ReferenceGenerator gen = new Gemma3ReferenceGenerator(randomWeights(c));
        int[] prompt = {2, 5, 9};

        AtomicInteger callbacks = new AtomicInteger();
        Gemma3ReferenceGenerator.Result r = gen.generate(prompt, 4, t -> callbacks.incrementAndGet());

        assertTrue(r.generatedTokenIds().length >= 1 && r.generatedTokenIds().length <= 4);
        assertEquals(r.generatedTokenIds().length, callbacks.get(), "callback once per generated token");
        for (int id : r.generatedTokenIds()) {
            assertTrue(id >= 0 && id < c.vocabSize());
        }
        // Determinism: a second run yields identical tokens.
        Gemma3ReferenceGenerator.Result r2 = gen.generate(prompt, 4);
        assertArrayEquals(r.generatedTokenIds(), r2.generatedTokenIds());
        // If it stopped early it must be because EOS was produced.
        if (r.generatedTokenIds().length < 4) {
            assertEquals(Gemma3ReferenceGenerator.FinishReason.EOS, r.finishReason());
            assertEquals(c.eosTokenId(), r.generatedTokenIds()[r.generatedTokenIds().length - 1]);
        }
    }
}
