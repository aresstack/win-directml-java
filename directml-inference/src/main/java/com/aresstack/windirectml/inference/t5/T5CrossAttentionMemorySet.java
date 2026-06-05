package com.aresstack.windirectml.inference.t5;

import java.util.List;
import java.util.Objects;

/**
 * Per-generation cross-attention memory for all decoder layers.
 */
public final class T5CrossAttentionMemorySet {
    private final T5EncoderOutput encoderOutput;
    private final List<T5CrossAttentionMemory> memories;

    public T5CrossAttentionMemorySet(T5EncoderOutput encoderOutput, List<T5CrossAttentionMemory> memories) {
        this.encoderOutput = Objects.requireNonNull(encoderOutput, "encoderOutput");
        this.memories = List.copyOf(Objects.requireNonNull(memories, "memories"));
        if (this.memories.isEmpty()) {
            throw new IllegalArgumentException("memories must not be empty");
        }
    }

    public boolean belongsTo(T5EncoderOutput candidate) {
        return encoderOutput == candidate;
    }

    public T5CrossAttentionMemory memory(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= memories.size()) {
            throw new IllegalArgumentException("layerIndex outside cross-attention memory set: " + layerIndex);
        }
        return memories.get(layerIndex);
    }

    public int layers() {
        return memories.size();
    }
}
