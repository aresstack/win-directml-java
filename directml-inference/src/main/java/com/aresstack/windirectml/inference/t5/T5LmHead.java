package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Reference LM head for T5 token generation.
 *
 * <p>This class exists only for the v38 correctness path. Production T5
 * execution must project logits through WARP instead of Java arrays.</p>
 */
public final class T5LmHead implements T5LogitProjector {
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

    @Override
    public float[] logits(float[] decoderHiddenState) {
        float[] logits = new float[vocabularySize()];
        logitsInto(decoderHiddenState, logits);
        return logits;
    }

    @Override
    public void logitsInto(float[] decoderHiddenState, float[] outputLogits) {
        Objects.requireNonNull(decoderHiddenState, "decoderHiddenState");
        Objects.requireNonNull(outputLogits, "outputLogits");
        if (decoderHiddenState.length != lmHeadWeight.dim(1)) {
            throw new IllegalArgumentException("Decoder hidden state length mismatch for T5 LM head: hidden="
                    + decoderHiddenState.length + ", expected=" + lmHeadWeight.dim(1));
        }
        if (outputLogits.length < vocabularySize()) {
            throw new IllegalArgumentException("T5 LM head output buffer too small: " + outputLogits.length
                    + " < " + vocabularySize());
        }
        T5ReferenceMath.denseInto(decoderHiddenState, lmHeadWeight, outputLogits);
    }

    @Override
    public int vocabularySize() {
        return lmHeadWeight.dim(0);
    }
}
