package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Reference T5 decoder cross-attention over encoder hidden states.
 */
public final class T5CrossAttention {
    private final T5PackageMetadata metadata;
    private final T5LinearProjection q;
    private final T5CrossAttentionMemoryProjection memoryProjection;
    private final T5LinearProjection o;

    public T5CrossAttention(T5PackageMetadata metadata,
                            T5TensorData q,
                            T5TensorData k,
                            T5TensorData v,
                            T5TensorData o) {
        this(metadata,
                T5ReferenceLinearProjection.from(q),
                new T5SplitCrossAttentionMemoryProjection(
                        T5ReferenceLinearProjection.from(k),
                        T5ReferenceLinearProjection.from(v)),
                T5ReferenceLinearProjection.from(o));
    }

    public T5CrossAttention(T5PackageMetadata metadata,
                            T5LinearProjection q,
                            T5LinearProjection k,
                            T5LinearProjection v,
                            T5LinearProjection o) {
        this(metadata,
                q,
                new T5SplitCrossAttentionMemoryProjection(k, v),
                o);
    }

    public T5CrossAttention(T5PackageMetadata metadata,
                            T5LinearProjection q,
                            T5CrossAttentionMemoryProjection memoryProjection,
                            T5LinearProjection o) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.q = Objects.requireNonNull(q, "q");
        this.memoryProjection = Objects.requireNonNull(memoryProjection, "memoryProjection");
        this.o = Objects.requireNonNull(o, "o");
    }

    public T5CrossAttentionMemory prepareMemory(T5EncoderOutput encoderOutput) {
        Objects.requireNonNull(encoderOutput, "encoderOutput");
        int hiddenSize = metadata.dModel();
        int heads = metadata.numHeads();
        int headDim = metadata.dKv();
        int innerSize = heads * headDim;
        if (encoderOutput.hiddenSize() != hiddenSize) {
            throw new IllegalArgumentException("Encoder hidden size mismatch: "
                    + encoderOutput.hiddenSize() + " != " + hiddenSize);
        }
        int encoderLength = encoderOutput.inputTokens();
        float[] encoderHiddenStates = encoderOutput.hiddenStates();
        T5ProjectedCrossAttentionMemory projected = memoryProjection.applySequence(
                encoderHiddenStates, encoderLength, hiddenSize);
        return new T5CrossAttentionMemory(encoderLength, innerSize, encoderOutput.attentionMask(),
                projected.key(), projected.value());
    }

    public float[] apply(float[] decoderHiddenStates, int decoderLength, T5EncoderOutput encoderOutput) {
        return apply(decoderHiddenStates, decoderLength, prepareMemory(encoderOutput));
    }

    public float[] apply(float[] decoderHiddenStates, int decoderLength, T5CrossAttentionMemory memory) {
        Objects.requireNonNull(memory, "memory");
        int hiddenSize = metadata.dModel();
        int heads = metadata.numHeads();
        int headDim = metadata.dKv();
        int innerSize = heads * headDim;
        validateHidden(decoderHiddenStates, decoderLength, hiddenSize);
        if (memory.innerSize() != innerSize) {
            throw new IllegalArgumentException("Cross-attention memory inner size mismatch: "
                    + memory.innerSize() + " != " + innerSize);
        }
        int encoderLength = memory.encoderLength();
        float[] query = q.applySequence(decoderHiddenStates, decoderLength, hiddenSize);
        float[] context = new float[decoderLength * innerSize];
        for (int token = 0; token < decoderLength; token++) {
            for (int head = 0; head < heads; head++) {
                float[] scores = new float[encoderLength];
                int headOffset = head * headDim;
                for (int source = 0; source < encoderLength; source++) {
                    if (!memory.attentionEnabled(source)) {
                        scores[source] = -1.0e9f;
                        continue;
                    }
                    float score = dot(query, memory, token, source, headOffset, innerSize, headDim);
                    scores[source] = score;
                }
                T5ReferenceMath.softmaxInPlace(scores);
                for (int dim = 0; dim < headDim; dim++) {
                    float sum = 0.0f;
                    for (int source = 0; source < encoderLength; source++) {
                        sum += scores[source] * memory.valueAt(source, headOffset, dim);
                    }
                    context[token * innerSize + headOffset + dim] = T5ReferenceMath.finite(sum);
                }
            }
        }
        return o.applySequence(context, decoderLength, innerSize);
    }

    private static float dot(float[] query, T5CrossAttentionMemory memory, int queryToken,
                             int keyToken, int headOffset, int innerSize, int headDim) {
        float sum = 0.0f;
        int queryOffset = queryToken * innerSize + headOffset;
        for (int dim = 0; dim < headDim; dim++) {
            sum += query[queryOffset + dim] * memory.keyAt(keyToken, headOffset, dim);
        }
        return T5ReferenceMath.finite(sum);
    }

    private static void validateHidden(float[] hiddenStates, int sequenceLength, int hiddenSize) {
        if (hiddenStates.length != sequenceLength * hiddenSize) {
            throw new IllegalArgumentException("decoderHiddenStates length mismatch: " + hiddenStates.length
                    + ", expected=" + (sequenceLength * hiddenSize));
        }
    }
}
