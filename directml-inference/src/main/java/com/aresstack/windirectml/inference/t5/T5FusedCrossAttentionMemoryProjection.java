package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Cross-attention memory K/V projection backed by one fused linear projection.
 */
public final class T5FusedCrossAttentionMemoryProjection implements T5CrossAttentionMemoryProjection {
    private final T5LinearProjection fusedProjection;
    private final int innerSize;

    public T5FusedCrossAttentionMemoryProjection(T5LinearProjection fusedProjection, int innerSize) {
        this.fusedProjection = Objects.requireNonNull(fusedProjection, "fusedProjection");
        this.innerSize = innerSize;
    }

    public static T5TensorData fuseWeights(String name, T5TensorData key, T5TensorData value) {
        validateCompatible(key, value);
        int rows = key.dim(0);
        int columns = key.dim(1);
        float[] fused = new float[rows * columns * 2];
        System.arraycopy(key.values(), 0, fused, 0, rows * columns);
        System.arraycopy(value.values(), 0, fused, rows * columns, rows * columns);
        return T5TensorData.reference(name, new long[]{rows * 2L, columns}, fused);
    }

    @Override
    public T5ProjectedCrossAttentionMemory applySequence(float[] encoderHiddenStates, int encoderLength, int hiddenSize) {
        float[] fused = fusedProjection.applySequence(encoderHiddenStates, encoderLength, hiddenSize);
        int expected = encoderLength * innerSize * 2;
        if (fused.length != expected) {
            throw new IllegalStateException("Fused cross-attention memory projection length mismatch: "
                    + fused.length + " != " + expected);
        }
        float[] key = new float[encoderLength * innerSize];
        float[] value = new float[encoderLength * innerSize];
        for (int token = 0; token < encoderLength; token++) {
            int fusedBase = token * innerSize * 2;
            int targetBase = token * innerSize;
            System.arraycopy(fused, fusedBase, key, targetBase, innerSize);
            System.arraycopy(fused, fusedBase + innerSize, value, targetBase, innerSize);
        }
        return new T5ProjectedCrossAttentionMemory(key, value);
    }

    private static void validateCompatible(T5TensorData key, T5TensorData value) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        if (key.rank() != 2 || value.rank() != 2) {
            throw new IllegalArgumentException("K/V weights must both be rank 2");
        }
        if (!Arrays.equals(key.dims(), value.dims())) {
            throw new IllegalArgumentException("K/V weights must share dimensions: k="
                    + Arrays.toString(key.dims()) + ", v=" + Arrays.toString(value.dims()));
        }
    }
}
