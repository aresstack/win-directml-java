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

    /**
     * Create the self-attention Q/K/V projection strategy.
     *
     * <p>The default implementation keeps three independent projections. WARP
     * factories may override this to fuse Q/K/V into one native dispatch.</p>
     *
     * @param queryWeight query projection weight
     * @param keyWeight key projection weight
     * @param valueWeight value projection weight
     * @return self-attention projection strategy
     */
    default T5SelfAttentionProjection createSelfAttentionProjection(T5TensorData queryWeight,
                                                                    T5TensorData keyWeight,
                                                                    T5TensorData valueWeight) {
        return new T5SplitSelfAttentionProjection(create(queryWeight), create(keyWeight), create(valueWeight));
    }

    /**
     * Create the cross-attention memory K/V projection strategy.
     *
     * <p>The default implementation keeps independent K and V projections. WARP
     * factories may override this to fuse K/V memory preparation into one native
     * dispatch.</p>
     *
     * @param keyWeight key projection weight
     * @param valueWeight value projection weight
     * @return cross-attention memory projection strategy
     */
    default T5CrossAttentionMemoryProjection createCrossAttentionMemoryProjection(T5TensorData keyWeight,
                                                                                  T5TensorData valueWeight) {
        return new T5SplitCrossAttentionMemoryProjection(create(keyWeight), create(valueWeight));
    }

    @Override
    default void close() {
        // Keep stateless factories no-op closeable.
    }
}
