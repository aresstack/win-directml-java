package com.aresstack.windirectml.inference.smollm2;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Per-layer key/value cache for the SmolLM2 reference decoder.
 */
public final class SmolLM2ReferenceLayerKvCache {

    private final int keyWidth;
    private final List<float[]> keys = new ArrayList<>();
    private final List<float[]> values = new ArrayList<>();

    SmolLM2ReferenceLayerKvCache(int keyWidth) {
        if (keyWidth <= 0) {
            throw new IllegalArgumentException("keyWidth must be > 0");
        }
        this.keyWidth = keyWidth;
    }

    public void append(float[] key, float[] value) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        if (key.length != keyWidth) {
            throw new IllegalArgumentException("key width mismatch: expected " + keyWidth + " but got " + key.length);
        }
        if (value.length != keyWidth) {
            throw new IllegalArgumentException("value width mismatch: expected " + keyWidth + " but got " + value.length);
        }
        keys.add(key.clone());
        values.add(value.clone());
    }

    public float[] keyAt(int position) {
        return keys.get(position);
    }

    public float[] valueAt(int position) {
        return values.get(position);
    }

    public int size() {
        return keys.size();
    }

    public int keyWidth() {
        return keyWidth;
    }
}
