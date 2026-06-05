package com.aresstack.windirectml.inference.t5;

/**
 * Creates reference Java projections.
 */
public final class T5ReferenceLinearProjectionFactory implements T5LinearProjectionFactory {
    public static final T5ReferenceLinearProjectionFactory INSTANCE = new T5ReferenceLinearProjectionFactory();

    private T5ReferenceLinearProjectionFactory() {
    }

    @Override
    public T5LinearProjection create(T5TensorData weight) {
        return T5ReferenceLinearProjection.from(weight);
    }
}
