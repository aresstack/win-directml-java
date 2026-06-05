package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Reference Java projection used by correctness paths and tests.
 */
public final class T5ReferenceLinearProjection implements T5LinearProjection {
    private final T5TensorData weight;

    private T5ReferenceLinearProjection(T5TensorData weight) {
        this.weight = Objects.requireNonNull(weight, "weight");
        if (weight.rank() != 2) {
            throw new IllegalArgumentException("Projection weight must be rank 2: " + weight.name());
        }
    }

    public static T5ReferenceLinearProjection from(T5TensorData weight) {
        return new T5ReferenceLinearProjection(weight);
    }

    @Override
    public String name() {
        return weight.name();
    }

    @Override
    public int inputSize() {
        return weight.dim(1);
    }

    @Override
    public int outputSize() {
        return weight.dim(0);
    }

    @Override
    public float[] apply(float[] input) {
        return T5ReferenceMath.dense(input, weight);
    }
}
