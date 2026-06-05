package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Reference T5 decoder cross-attention over encoder hidden states.
 */
public final class T5CrossAttention {
    private final T5PackageMetadata metadata;
    private final T5LinearProjection q;
    private final T5LinearProjection k;
    private final T5LinearProjection v;
    private final T5LinearProjection o;

    public T5CrossAttention(T5PackageMetadata metadata,
                            T5TensorData q,
                            T5TensorData k,
                            T5TensorData v,
                            T5TensorData o) {
        this(metadata,
                T5ReferenceLinearProjection.from(q),
                T5ReferenceLinearProjection.from(k),
                T5ReferenceLinearProjection.from(v),
                T5ReferenceLinearProjection.from(o));
    }

    public T5CrossAttention(T5PackageMetadata metadata,
                            T5LinearProjection q,
                            T5LinearProjection k,
                            T5LinearProjection v,
                            T5LinearProjection o) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.q = Objects.requireNonNull(q, "q");
        this.k = Objects.requireNonNull(k, "k");
        this.v = Objects.requireNonNull(v, "v");
        this.o = Objects.requireNonNull(o, "o");
    }

    public float[] apply(float[] decoderHiddenStates, int decoderLength, T5EncoderOutput encoderOutput) {
        Objects.requireNonNull(encoderOutput, "encoderOutput");
        int hiddenSize = metadata.dModel();
        int heads = metadata.numHeads();
        int headDim = metadata.dKv();
        int innerSize = heads * headDim;
        validateHidden(decoderHiddenStates, decoderLength, hiddenSize);
        if (encoderOutput.hiddenSize() != hiddenSize) {
            throw new IllegalArgumentException("Encoder hidden size mismatch: "
                    + encoderOutput.hiddenSize() + " != " + hiddenSize);
        }
        int encoderLength = encoderOutput.inputTokens();
        boolean[] encoderMask = encoderOutput.attentionMask();
        float[] encoderHiddenStates = encoderOutput.hiddenStates();
        float[] query = q.applySequence(decoderHiddenStates, decoderLength, hiddenSize);
        float[] key = k.applySequence(encoderHiddenStates, encoderLength, hiddenSize);
        float[] value = v.applySequence(encoderHiddenStates, encoderLength, hiddenSize);
        float[] context = new float[decoderLength * innerSize];
        for (int token = 0; token < decoderLength; token++) {
            for (int head = 0; head < heads; head++) {
                float[] scores = new float[encoderLength];
                for (int source = 0; source < encoderLength; source++) {
                    if (!encoderMask[source]) {
                        scores[source] = -1.0e9f;
                        continue;
                    }
                    float score = dot(query, key, token, source, head, innerSize, headDim);
                    scores[source] = score;
                }
                T5ReferenceMath.softmaxInPlace(scores);
                for (int dim = 0; dim < headDim; dim++) {
                    float sum = 0.0f;
                    for (int source = 0; source < encoderLength; source++) {
                        int valueIndex = source * innerSize + head * headDim + dim;
                        sum += scores[source] * value[valueIndex];
                    }
                    context[token * innerSize + head * headDim + dim] = T5ReferenceMath.finite(sum);
                }
            }
        }
        return o.applySequence(context, decoderLength, innerSize);
    }

    private static float dot(float[] query, float[] key, int queryToken, int keyToken, int head, int innerSize, int headDim) {
        float sum = 0.0f;
        int queryOffset = queryToken * innerSize + head * headDim;
        int keyOffset = keyToken * innerSize + head * headDim;
        for (int dim = 0; dim < headDim; dim++) {
            sum += query[queryOffset + dim] * key[keyOffset + dim];
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
