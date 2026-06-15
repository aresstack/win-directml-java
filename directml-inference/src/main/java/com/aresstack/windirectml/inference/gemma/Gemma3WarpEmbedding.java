package com.aresstack.windirectml.inference.gemma;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Gemma 3 input embedding lookup for the WARP prefill (GEMMA-WARP-9): {@code hidden[t] =
 * embed_tokens[id] * sqrt(hidden_size)}.
 *
 * <p>Heap-light by construction: only the prompt's token rows are read (a handful), never the whole
 * {@code vocab*hidden} matrix. The same tied {@code embed_tokens} matrix also backs the LM head
 * ({@link Gemma3WarpLmHead}), so it is materialised/uploaded once — not twice. Two row sources are
 * supported: a host {@code float[]} (reference/synthetic) and a row-major FP32 {@link ByteBuffer}
 * (heap-light, e.g. a decoded SafeTensors payload).</p>
 */
public final class Gemma3WarpEmbedding {

    private Gemma3WarpEmbedding() {
    }

    /** Lookup + scale from a {@code [vocab, hidden]} row-major {@code float[]}. */
    public static float[][] lookupScaled(float[] embedding, int[] ids, int hidden, float scale) {
        float[][] out = new float[ids.length][hidden];
        for (int t = 0; t < ids.length; t++) {
            int base = Math.multiplyExact(ids[t], hidden);
            if (base < 0 || base + hidden > embedding.length) {
                throw new IllegalArgumentException("token id out of range: " + ids[t]);
            }
            for (int i = 0; i < hidden; i++) {
                out[t][i] = embedding[base + i] * scale;
            }
        }
        return out;
    }

    /** Lookup + scale from a {@code [vocab, hidden]} row-major little-endian FP32 {@link ByteBuffer}. */
    public static float[][] lookupScaled(ByteBuffer embeddingFp32Le, int[] ids, int hidden, float scale) {
        if (embeddingFp32Le.order() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalArgumentException("embedding ByteBuffer must be LITTLE_ENDIAN");
        }
        float[][] out = new float[ids.length][hidden];
        for (int t = 0; t < ids.length; t++) {
            long byteBase = (long) ids[t] * hidden * Float.BYTES;
            if (ids[t] < 0 || byteBase + (long) hidden * Float.BYTES > embeddingFp32Le.limit()) {
                throw new IllegalArgumentException("token id out of range: " + ids[t]);
            }
            for (int i = 0; i < hidden; i++) {
                out[t][i] = embeddingFp32Le.getFloat((int) (byteBase + (long) i * Float.BYTES)) * scale;
            }
        }
        return out;
    }
}
