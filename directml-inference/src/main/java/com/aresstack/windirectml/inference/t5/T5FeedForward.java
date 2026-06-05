package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Reference T5 feed-forward block for dense and gated projections.
 */
public final class T5FeedForward {
    private final T5PackageMetadata metadata;
    private final T5LinearProjection wi;
    private final T5LinearProjection wi0;
    private final T5LinearProjection wi1;
    private final T5LinearProjection wo;

    public T5FeedForward(T5PackageMetadata metadata, T5TensorData wi, T5TensorData wi0, T5TensorData wi1, T5TensorData wo) {
        this(metadata,
                wi == null ? null : T5ReferenceLinearProjection.from(wi),
                wi0 == null ? null : T5ReferenceLinearProjection.from(wi0),
                wi1 == null ? null : T5ReferenceLinearProjection.from(wi1),
                T5ReferenceLinearProjection.from(wo));
    }

    public T5FeedForward(T5PackageMetadata metadata, T5LinearProjection wi, T5LinearProjection wi0,
                         T5LinearProjection wi1, T5LinearProjection wo) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.wi = wi;
        this.wi0 = wi0;
        this.wi1 = wi1;
        this.wo = Objects.requireNonNull(wo, "wo");
    }

    public float[] apply(float[] hiddenStates, int sequenceLength) {
        int hiddenSize = metadata.dModel();
        float[] result = new float[hiddenStates.length];
        for (int token = 0; token < sequenceLength; token++) {
            float[] input = T5ReferenceMath.slice(hiddenStates, token * hiddenSize, hiddenSize);
            float[] output = applyToken(input);
            T5ReferenceMath.copyInto(output, result, token * hiddenSize);
        }
        return result;
    }

    private float[] applyToken(float[] input) {
        if (isGated()) {
            float[] gate = wi0.apply(input);
            float[] values = wi1.apply(input);
            float[] activated = new float[gate.length];
            for (int i = 0; i < activated.length; i++) {
                activated[i] = activation(gate[i]) * values[i];
            }
            return wo.apply(activated);
        }
        float[] inner = wi.apply(input);
        for (int i = 0; i < inner.length; i++) {
            inner[i] = activation(inner[i]);
        }
        return wo.apply(inner);
    }

    private boolean isGated() {
        return wi0 != null && wi1 != null;
    }

    private float activation(float value) {
        String projection = metadata.feedForwardProjection();
        if (projection != null && projection.contains("gelu")) {
            return T5ReferenceMath.gelu(value);
        }
        return T5ReferenceMath.relu(value);
    }
}
