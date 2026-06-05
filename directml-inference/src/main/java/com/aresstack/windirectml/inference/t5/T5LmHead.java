package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Reference LM head for T5 token generation.
 *
 * <p>This class exists only for the v38 correctness path. Production T5
 * execution must project logits through WARP instead of Java arrays.</p>
 */
public final class T5LmHead {
    private final T5TensorData lmHeadWeight;

    private T5LmHead(T5TensorData lmHeadWeight) {
        this.lmHeadWeight = Objects.requireNonNull(lmHeadWeight, "lmHeadWeight");
        if (lmHeadWeight.rank() != 2) {
            throw new IllegalArgumentException("T5 LM head must be rank 2: " + lmHeadWeight.name());
        }
    }

    public static T5LmHead from(T5Weights weights) {
        Objects.requireNonNull(weights, "weights");
        return new T5LmHead(T5TensorData.from(weights.lmHead()));
    }

    public float[] logits(float[] decoderHiddenState) {
        Objects.requireNonNull(decoderHiddenState, "decoderHiddenState");
        if (decoderHiddenState.length != lmHeadWeight.dim(1)) {
            throw new IllegalArgumentException("Decoder hidden state length mismatch for T5 LM head: hidden="
                    + decoderHiddenState.length + ", expected=" + lmHeadWeight.dim(1));
        }
        return T5ReferenceMath.dense(decoderHiddenState, lmHeadWeight);
    }

    public int vocabularySize() {
        return lmHeadWeight.dim(0);
    }
}
