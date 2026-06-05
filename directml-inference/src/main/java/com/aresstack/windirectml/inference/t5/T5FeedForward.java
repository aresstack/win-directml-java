package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * T5 feed-forward block for dense and gated projections.
 *
 * <p>The implementation is sequence-oriented on purpose. Encoder prefill and
 * decoder-prefix execution must use {@link T5LinearProjection#applySequence}
 * so WARP-backed projections can submit one batched matrix multiplication per
 * projection instead of one dispatch per token. This keeps the reference path
 * simple and gives the native path a useful batch boundary.</p>
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
        Objects.requireNonNull(hiddenStates, "hiddenStates");
        int hiddenSize = metadata.dModel();
        if (hiddenStates.length != sequenceLength * hiddenSize) {
            throw new IllegalArgumentException("T5 feed-forward hidden length mismatch: " + hiddenStates.length
                    + ", expected=" + (sequenceLength * hiddenSize));
        }
        if (isGated()) {
            return applyGatedSequence(hiddenStates, sequenceLength, hiddenSize);
        }
        return applyDenseSequence(hiddenStates, sequenceLength, hiddenSize);
    }

    private float[] applyDenseSequence(float[] hiddenStates, int sequenceLength, int hiddenSize) {
        if (wi == null) {
            throw new IllegalStateException("T5 feed-forward dense wi projection is missing");
        }
        float[] inner = wi.applySequence(hiddenStates, sequenceLength, hiddenSize);
        activateInPlace(inner);
        return wo.applySequence(inner, sequenceLength, wi.outputSize());
    }

    private float[] applyGatedSequence(float[] hiddenStates, int sequenceLength, int hiddenSize) {
        float[] gate = wi0.applySequence(hiddenStates, sequenceLength, hiddenSize);
        float[] values = wi1.applySequence(hiddenStates, sequenceLength, hiddenSize);
        if (gate.length != values.length) {
            throw new IllegalStateException("T5 gated feed-forward projection size mismatch: gate="
                    + gate.length + ", values=" + values.length);
        }
        for (int i = 0; i < gate.length; i++) {
            gate[i] = T5ReferenceMath.finite(activation(gate[i]) * values[i]);
        }
        return wo.applySequence(gate, sequenceLength, wi0.outputSize());
    }

    private void activateInPlace(float[] values) {
        for (int i = 0; i < values.length; i++) {
            values[i] = activation(values[i]);
        }
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
