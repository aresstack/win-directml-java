package com.aresstack.windirectml.inference.t5;

/**
 * Projects encoder hidden states into decoder cross-attention K/V memory.
 */
public interface T5CrossAttentionMemoryProjection {
    T5ProjectedCrossAttentionMemory applySequence(float[] encoderHiddenStates, int encoderLength, int hiddenSize);
}
