package com.aresstack.windirectml.inference.decoderonly;

import java.util.Objects;

/**
 * Per-layer key/value cache for the decoder-only WARP path, backed by contiguous {@code float[]} buffers instead of a
 * {@code List<float[]>} of per-token rows.
 *
 * <p>Keys and values for token {@code p} live at {@code [p * keyWidth .. (p+1) * keyWidth)} in two pre-sized arrays.
 * This keeps attention reads cache-friendly and avoids per-token array allocations. The stored data and the resulting
 * attention math are family-neutral; both Qwen-like and Llama-like (SmolLM2) layouts share this cache.</p>
 */
public final class DecoderOnlyWarpLayerKvCache {

    private final int keyWidth;
    private final int capacity;
    private final float[] keys;
    private final float[] values;
    private int size;

    public DecoderOnlyWarpLayerKvCache(int keyWidth, int capacity) {
        if (keyWidth <= 0) {
            throw new IllegalArgumentException("keyWidth must be > 0");
        }
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be > 0");
        }
        this.keyWidth = keyWidth;
        this.capacity = capacity;
        this.keys = new float[Math.multiplyExact(keyWidth, capacity)];
        this.values = new float[Math.multiplyExact(keyWidth, capacity)];
    }

    /** Append a key/value pair read from {@code keyWidth}-wide slices of the given source buffers. */
    public void append(float[] keySource, int keyOffset, float[] valueSource, int valueOffset) {
        Objects.requireNonNull(keySource, "keySource");
        Objects.requireNonNull(valueSource, "valueSource");
        if (size >= capacity) {
            throw new IllegalStateException("WARP KV cache layer is full: capacity=" + capacity);
        }
        int destination = size * keyWidth;
        System.arraycopy(keySource, keyOffset, keys, destination, keyWidth);
        System.arraycopy(valueSource, valueOffset, values, destination, keyWidth);
        size++;
    }

    /** Convenience append from full-width key/value arrays. */
    public void append(float[] key, float[] value) {
        if (key.length != keyWidth || value.length != keyWidth) {
            throw new IllegalArgumentException("key/value width mismatch: expected " + keyWidth);
        }
        append(key, 0, value, 0);
    }

    public float[] keys() {
        return keys;
    }

    public float[] values() {
        return values;
    }

    public int keyWidth() {
        return keyWidth;
    }

    public int size() {
        return size;
    }
}
