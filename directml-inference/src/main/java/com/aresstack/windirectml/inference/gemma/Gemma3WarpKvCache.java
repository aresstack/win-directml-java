package com.aresstack.windirectml.inference.gemma;

import java.util.Arrays;

/**
 * Per-layer key/value cache for the Gemma 3 WARP decode session (GEMMA-WARP-10a).
 *
 * <p>Holds the RoPE'd, QK-normed {@code k}/{@code v} for every processed position, per layer, as flat
 * row-major {@code [position][kvDim]} buffers ({@code kvDim = numKvHeads * head_dim}; Gemma 270M = 1×256).
 * The decode step appends one position per layer and reads back the visible range. This is a host-side
 * cache (the building-block kernels already round-trip through the CPU) and stores the <b>full</b> history
 * for every layer; the local/global sliding-window restriction is applied at <i>read</i> time by
 * {@link Gemma3AttentionLayout#firstValidKey} (so full layers see everything and local layers only their
 * window) rather than by evicting cache rows — correctness first, windowed eviction is a later perf
 * concern.</p>
 */
public final class Gemma3WarpKvCache {

    private final int numLayers;
    private final int kvDim;
    private float[][] k; // [layer][capacity * kvDim]
    private float[][] v;
    private int capacity;
    private int length;  // committed positions (advances once all layers wrote a token)

    public Gemma3WarpKvCache(int numLayers, int kvDim, int initialCapacity) {
        if (numLayers < 1 || kvDim < 1) {
            throw new IllegalArgumentException("numLayers and kvDim must be positive");
        }
        this.numLayers = numLayers;
        this.kvDim = kvDim;
        this.capacity = Math.max(1, initialCapacity);
        this.k = new float[numLayers][capacity * kvDim];
        this.v = new float[numLayers][capacity * kvDim];
        this.length = 0;
    }

    public int length() {
        return length;
    }

    public int kvDim() {
        return kvDim;
    }

    public int numLayers() {
        return numLayers;
    }

    /** Store {@code k}/{@code v} ({@code kvDim} each) for {@code layer} at position {@code pos}. */
    public void put(int layer, int pos, float[] kRow, float[] vRow) {
        if (kRow.length != kvDim || vRow.length != kvDim) {
            throw new IllegalArgumentException("k/v row length must equal kvDim=" + kvDim);
        }
        ensureCapacity(pos + 1);
        System.arraycopy(kRow, 0, k[layer], pos * kvDim, kvDim);
        System.arraycopy(vRow, 0, v[layer], pos * kvDim, kvDim);
    }

    /** Mark {@code newLength} positions as committed (called once a token has been written to all layers). */
    public void commitLength(int newLength) {
        this.length = newLength;
    }

    /** A copy of {@code layer}'s keys for positions {@code [0, len)} as a {@code len*kvDim} flat buffer. */
    public float[] kFlat(int layer, int len) {
        return Arrays.copyOf(k[layer], len * kvDim);
    }

    /** A copy of {@code layer}'s values for positions {@code [0, len)} as a {@code len*kvDim} flat buffer. */
    public float[] vFlat(int layer, int len) {
        return Arrays.copyOf(v[layer], len * kvDim);
    }

    public void reset() {
        length = 0;
    }

    private void ensureCapacity(int need) {
        if (need <= capacity) {
            return;
        }
        int newCap = capacity;
        while (newCap < need) {
            newCap *= 2;
        }
        for (int l = 0; l < numLayers; l++) {
            k[l] = Arrays.copyOf(k[l], newCap * kvDim);
            v[l] = Arrays.copyOf(v[l], newCap * kvDim);
        }
        capacity = newCap;
    }
}
