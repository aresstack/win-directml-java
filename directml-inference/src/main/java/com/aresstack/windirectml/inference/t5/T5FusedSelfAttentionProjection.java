package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Self-attention Q/K/V projection backed by one fused linear projection.
 *
 * <p>The fused projection keeps the T5 runtime model-specific, but reduces WARP
 * dispatch count from three Q/K/V matvecs to one matvec for each self-attention
 * invocation.</p>
 */
public final class T5FusedSelfAttentionProjection implements T5SelfAttentionProjection {
    private final T5LinearProjection fusedProjection;
    private final int innerSize;

    public T5FusedSelfAttentionProjection(T5LinearProjection fusedProjection, int innerSize) {
        this.fusedProjection = Objects.requireNonNull(fusedProjection, "fusedProjection");
        this.innerSize = innerSize;
    }

    public static T5TensorData fuseWeights(String name, T5TensorData query, T5TensorData key, T5TensorData value) {
        validateCompatible(query, key, value);
        int rows = query.dim(0);
        int columns = query.dim(1);
        float[] fused = new float[rows * columns * 3];
        appendRows(fused, 0, query.values());
        appendRows(fused, rows * columns, key.values());
        appendRows(fused, rows * columns * 2, value.values());
        return T5TensorData.reference(name, new long[]{rows * 3L, columns}, fused);
    }

    @Override
    public T5ProjectedSelfAttention applySequence(float[] hiddenStates, int sequenceLength, int hiddenSize) {
        float[] fused = fusedProjection.applySequence(hiddenStates, sequenceLength, hiddenSize);
        return split(fused, sequenceLength);
    }

    @Override
    public T5ProjectedSelfAttention apply(float[] hiddenState) {
        float[] fused = fusedProjection.apply(hiddenState);
        return split(fused, 1);
    }

    private T5ProjectedSelfAttention split(float[] fused, int sequenceLength) {
        int expected = sequenceLength * innerSize * 3;
        if (fused.length != expected) {
            throw new IllegalStateException("Fused self-attention projection length mismatch: "
                    + fused.length + " != " + expected);
        }
        float[] query = new float[sequenceLength * innerSize];
        float[] key = new float[sequenceLength * innerSize];
        float[] value = new float[sequenceLength * innerSize];
        for (int token = 0; token < sequenceLength; token++) {
            int fusedBase = token * innerSize * 3;
            int targetBase = token * innerSize;
            System.arraycopy(fused, fusedBase, query, targetBase, innerSize);
            System.arraycopy(fused, fusedBase + innerSize, key, targetBase, innerSize);
            System.arraycopy(fused, fusedBase + innerSize * 2, value, targetBase, innerSize);
        }
        return new T5ProjectedSelfAttention(query, key, value);
    }

    private static void validateCompatible(T5TensorData query, T5TensorData key, T5TensorData value) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        if (query.rank() != 2 || key.rank() != 2 || value.rank() != 2) {
            throw new IllegalArgumentException("Q/K/V weights must all be rank 2");
        }
        if (!Arrays.equals(query.dims(), key.dims()) || !Arrays.equals(query.dims(), value.dims())) {
            throw new IllegalArgumentException("Q/K/V weights must share dimensions: q="
                    + Arrays.toString(query.dims()) + ", k=" + Arrays.toString(key.dims())
                    + ", v=" + Arrays.toString(value.dims()));
        }
    }

    private static void appendRows(float[] target, int targetOffset, float[] values) {
        System.arraycopy(values, 0, target, targetOffset, values.length);
    }
}
