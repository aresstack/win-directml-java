package com.aresstack.windirectml.inference.t5;

/**
 * Factory for T5 linear projections.
 */
public interface T5LinearProjectionFactory extends AutoCloseable {
    /**
     * Create a projection for the supplied tensor.
     *
     * @param weight projection weight
     * @return projection implementation
     */
    T5LinearProjection create(T5TensorData weight);

    @Override
    default void close() {
        // Keep stateless factories no-op closeable.
    }
}
