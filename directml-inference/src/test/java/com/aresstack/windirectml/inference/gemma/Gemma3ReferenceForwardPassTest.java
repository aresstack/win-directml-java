package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.model.SafeTensorsReader.SafeTensorEntry;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** GEMMA-WARP (reference): the full forward pass runs on a tiny synthetic Gemma; BF16 decode round-trips. */
class Gemma3ReferenceForwardPassTest {

    private static Gemma3Config tinyConfig() {
        // H=8, heads=2, kv=1, head_dim=4 -> attnDim=8, kvDim=4; inter=16; vocab=20; 2 layers; window=3.
        return new Gemma3Config("gemma3_text", List.of("Gemma3ForCausalLM"),
                8, 16, 2, 2, 1, 4, 20, 64,
                1e-6, 1_000_000, 10_000, 3, 0,
                List.of(Gemma3Config.SLIDING_ATTENTION, Gemma3Config.FULL_ATTENTION),
                4, "gelu_pytorch_tanh", 2, 1, 0, true);
    }

    private static Gemma3ReferenceWeights randomWeights(Gemma3Config c) {
        Random rng = new Random(42);
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
    void forwardProducesFiniteVocabLogitsAndIsDeterministic() {
        Gemma3Config c = tinyConfig();
        Gemma3ReferenceForwardPass fp = new Gemma3ReferenceForwardPass(randomWeights(c));
        int[] tokens = {2, 5, 7, 1, 9}; // length 5 > sliding_window 3 -> exercises the window on layer 0
        float[] logits = fp.logitsForLastToken(tokens);
        assertEquals(c.vocabSize(), logits.length);
        for (float v : logits) {
            assertTrue(Float.isFinite(v), "logit must be finite");
        }
        assertArrayEquals(logits, fp.logitsForLastToken(tokens), 0f, "forward pass must be deterministic");
        int next = fp.nextToken(tokens);
        assertTrue(next >= 0 && next < c.vocabSize());
    }

    @Test
    void singleTokenPrefillWorks() {
        Gemma3Config c = tinyConfig();
        Gemma3ReferenceForwardPass fp = new Gemma3ReferenceForwardPass(randomWeights(c));
        float[] logits = fp.logitsForLastToken(new int[]{2});
        assertEquals(c.vocabSize(), logits.length);
        for (float v : logits) {
            assertTrue(Float.isFinite(v));
        }
    }

    @Test
    void bf16DecodeRoundTrips() throws Exception {
        // BF16 of 1.0 = 0x3F80, 2.0 = 0x4000, -1.5 = 0xBFC0.
        ByteBuffer buf = ByteBuffer.allocate(6).order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort((short) 0x3F80);
        buf.putShort((short) 0x4000);
        buf.putShort((short) 0xBFC0);
        buf.flip();
        SafeTensorEntry entry = new SafeTensorEntry("t", "BF16", 16, new long[]{3}, 0, 6, 6, buf);
        float[] out = Gemma3ReferenceWeights.decodeFloats(entry);
        assertArrayEquals(new float[]{1.0f, 2.0f, -1.5f}, out, 0f);
    }
}
