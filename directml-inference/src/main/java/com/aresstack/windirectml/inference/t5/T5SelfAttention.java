package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Reference T5 self-attention used to validate encoder tensor layout and math.
 */
public final class T5SelfAttention {
    private final T5PackageMetadata metadata;
    private final T5LinearProjection q;
    private final T5LinearProjection k;
    private final T5LinearProjection v;
    private final T5LinearProjection o;
    private final T5RelativePositionBias relativePositionBias;
    private final boolean bidirectionalRelativeBias;
    private final boolean causalMask;

    public T5SelfAttention(T5PackageMetadata metadata,
                           T5TensorData q,
                           T5TensorData k,
                           T5TensorData v,
                           T5TensorData o,
                           T5RelativePositionBias relativePositionBias) {
        this(metadata,
                T5ReferenceLinearProjection.from(q),
                T5ReferenceLinearProjection.from(k),
                T5ReferenceLinearProjection.from(v),
                T5ReferenceLinearProjection.from(o),
                relativePositionBias, true, false);
    }

    public T5SelfAttention(T5PackageMetadata metadata,
                           T5TensorData q,
                           T5TensorData k,
                           T5TensorData v,
                           T5TensorData o,
                           T5RelativePositionBias relativePositionBias,
                           boolean bidirectionalRelativeBias,
                           boolean causalMask) {
        this(metadata,
                T5ReferenceLinearProjection.from(q),
                T5ReferenceLinearProjection.from(k),
                T5ReferenceLinearProjection.from(v),
                T5ReferenceLinearProjection.from(o),
                relativePositionBias, bidirectionalRelativeBias, causalMask);
    }

    public T5SelfAttention(T5PackageMetadata metadata,
                           T5LinearProjection q,
                           T5LinearProjection k,
                           T5LinearProjection v,
                           T5LinearProjection o,
                           T5RelativePositionBias relativePositionBias,
                           boolean bidirectionalRelativeBias,
                           boolean causalMask) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.q = Objects.requireNonNull(q, "q");
        this.k = Objects.requireNonNull(k, "k");
        this.v = Objects.requireNonNull(v, "v");
        this.o = Objects.requireNonNull(o, "o");
        this.relativePositionBias = relativePositionBias;
        this.bidirectionalRelativeBias = bidirectionalRelativeBias;
        this.causalMask = causalMask;
    }

    public float[] apply(float[] hiddenStates, int sequenceLength, boolean[] attentionMask) {
        int hiddenSize = metadata.dModel();
        int heads = metadata.numHeads();
        int headDim = metadata.dKv();
        int innerSize = heads * headDim;
        validateHidden(hiddenStates, sequenceLength, hiddenSize);
        boolean[] mask = normalizeMask(attentionMask, sequenceLength);
        float[] query = q.applySequence(hiddenStates, sequenceLength, hiddenSize);
        float[] key = k.applySequence(hiddenStates, sequenceLength, hiddenSize);
        float[] value = v.applySequence(hiddenStates, sequenceLength, hiddenSize);
        float[] context = new float[sequenceLength * innerSize];
        for (int token = 0; token < sequenceLength; token++) {
            if (!mask[token]) {
                continue;
            }
            for (int head = 0; head < heads; head++) {
                float[] scores = new float[sequenceLength];
                for (int source = 0; source < sequenceLength; source++) {
                    if (!mask[source] || (causalMask && source > token)) {
                        scores[source] = -1.0e9f;
                        continue;
                    }
                    float score = dot(query, key, token, source, head, innerSize, headDim);
                    if (relativePositionBias != null) {
                        score += relativePositionBias.value(head, token, source, bidirectionalRelativeBias);
                    }
                    scores[source] = score;
                }
                T5ReferenceMath.softmaxInPlace(scores);
                for (int dim = 0; dim < headDim; dim++) {
                    float sum = 0.0f;
                    for (int source = 0; source < sequenceLength; source++) {
                        int valueIndex = source * innerSize + head * headDim + dim;
                        sum += scores[source] * value[valueIndex];
                    }
                    context[token * innerSize + head * headDim + dim] = T5ReferenceMath.finite(sum);
                }
            }
        }
        float[] projected = o.applySequence(context, sequenceLength, innerSize);
        clearMaskedTokens(projected, sequenceLength, hiddenSize, mask);
        return projected;
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

    public T5SelfAttentionStep applyStep(float[] hiddenState, int tokenIndex, T5SelfAttentionMemory previousMemory) {
        int hiddenSize = metadata.dModel();
        int heads = metadata.numHeads();
        int headDim = metadata.dKv();
        int innerSize = heads * headDim;
        Objects.requireNonNull(hiddenState, "hiddenState");
        T5SelfAttentionMemory safeMemory = previousMemory == null ? T5SelfAttentionMemory.empty() : previousMemory;
        if (hiddenState.length != hiddenSize) {
            throw new IllegalArgumentException("hiddenState length mismatch: " + hiddenState.length
                    + ", expected=" + hiddenSize);
        }
        if (safeMemory.sequenceLength() != tokenIndex) {
            throw new IllegalArgumentException("Self-attention memory length mismatch: memory="
                    + safeMemory.sequenceLength() + ", tokenIndex=" + tokenIndex);
        }
        if (safeMemory.sequenceLength() > 0 && safeMemory.innerSize() != innerSize) {
            throw new IllegalArgumentException("Self-attention memory inner size mismatch: "
                    + safeMemory.innerSize() + " != " + innerSize);
        }
        float[] query = q.apply(hiddenState);
        float[] tokenKey = k.apply(hiddenState);
        float[] tokenValue = v.apply(hiddenState);
        T5SelfAttentionMemory memory = safeMemory.append(tokenKey, tokenValue);
        int totalLength = memory.sequenceLength();
        float[] context = new float[innerSize];
        for (int head = 0; head < heads; head++) {
            float[] scores = new float[totalLength];
            int headOffset = head * headDim;
            for (int source = 0; source < totalLength; source++) {
                float score = dot(query, memory, source, headOffset, headDim);
                if (relativePositionBias != null) {
                    score += relativePositionBias.value(head, tokenIndex, source, bidirectionalRelativeBias);
                }
                scores[source] = score;
            }
            T5ReferenceMath.softmaxInPlace(scores);
            for (int dim = 0; dim < headDim; dim++) {
                float sum = 0.0f;
                for (int source = 0; source < totalLength; source++) {
                    sum += scores[source] * memory.valueAt(source, headOffset, dim);
                }
                context[headOffset + dim] = T5ReferenceMath.finite(sum);
            }
        }
        return new T5SelfAttentionStep(o.apply(context), memory);
    }

    private static float dot(float[] query, T5SelfAttentionMemory memory, int keyToken,
                             int headOffset, int headDim) {
        float sum = 0.0f;
        for (int dim = 0; dim < headDim; dim++) {
            sum += query[headOffset + dim] * memory.keyAt(keyToken, headOffset, dim);
        }
        return T5ReferenceMath.finite(sum);
    }

    private static boolean[] normalizeMask(boolean[] mask, int sequenceLength) {
        if (mask == null) {
            boolean[] all = new boolean[sequenceLength];
            Arrays.fill(all, true);
            return all;
        }
        if (mask.length != sequenceLength) {
            throw new IllegalArgumentException("attentionMask length mismatch: " + mask.length + " != " + sequenceLength);
        }
        return Arrays.copyOf(mask, mask.length);
    }

    private static void validateHidden(float[] hiddenStates, int sequenceLength, int hiddenSize) {
        if (hiddenStates.length != sequenceLength * hiddenSize) {
            throw new IllegalArgumentException("hiddenStates length mismatch: " + hiddenStates.length
                    + ", expected=" + (sequenceLength * hiddenSize));
        }
    }

    private static void clearMaskedTokens(float[] values, int sequenceLength, int hiddenSize, boolean[] mask) {
        for (int token = 0; token < sequenceLength; token++) {
            if (!mask[token]) {
                Arrays.fill(values, token * hiddenSize, token * hiddenSize + hiddenSize, 0.0f);
            }
        }
    }
}
