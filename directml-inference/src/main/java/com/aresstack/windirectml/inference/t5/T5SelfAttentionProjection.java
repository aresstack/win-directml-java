package com.aresstack.windirectml.inference.t5;

/**
 * Projects T5 hidden states into self-attention Q/K/V tensors.
 */
public interface T5SelfAttentionProjection {
    /**
     * Project a full sequence.
     *
     * @param hiddenStates   hidden states [sequenceLength, hiddenSize]
     * @param sequenceLength sequence length
     * @param hiddenSize     hidden size
     * @return projected query, key and value tensors
     */
    T5ProjectedSelfAttention applySequence(float[] hiddenStates, int sequenceLength, int hiddenSize);

    /**
     * Project a single decoder token.
     *
     * @param hiddenState hidden state [hiddenSize]
     * @return projected query, key and value tensors for one token
     */
    T5ProjectedSelfAttention apply(float[] hiddenState);
}
