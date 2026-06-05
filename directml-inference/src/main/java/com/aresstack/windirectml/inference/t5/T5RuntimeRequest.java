package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Objects;

/**
 * Request boundary for the future T5 WARP generation runtime.
 */
public final class T5RuntimeRequest {
    private final int[] inputTokenIds;
    private final int maxNewTokens;
    private final T5StopTokenPolicy stopTokenPolicy;
    private final int decoderStartTokenId;
    private final float temperature;
    private final int topK;

    public T5RuntimeRequest(int[] inputTokenIds,
                            int maxNewTokens,
                            T5StopTokenPolicy stopTokenPolicy,
                            int decoderStartTokenId,
                            float temperature,
                            int topK) {
        this.inputTokenIds = copyInputTokens(inputTokenIds);
        this.maxNewTokens = requirePositive("maxNewTokens", maxNewTokens);
        this.stopTokenPolicy = Objects.requireNonNull(stopTokenPolicy, "stopTokenPolicy");
        if (decoderStartTokenId < 0) {
            throw new IllegalArgumentException("decoderStartTokenId must not be negative: " + decoderStartTokenId);
        }
        this.decoderStartTokenId = decoderStartTokenId;
        this.temperature = temperature;
        this.topK = topK;
    }

    public static T5RuntimeRequest greedy(int[] inputTokenIds, int maxNewTokens, T5SpecialTokens specialTokens) {
        Objects.requireNonNull(specialTokens, "specialTokens");
        return new T5RuntimeRequest(inputTokenIds, maxNewTokens, specialTokens.stopAtEos(),
                specialTokens.decoderStartTokenId(), 0.0f, 0);
    }

    public int[] inputTokenIds() {
        return Arrays.copyOf(inputTokenIds, inputTokenIds.length);
    }

    public int maxNewTokens() {
        return maxNewTokens;
    }

    public T5StopTokenPolicy stopTokenPolicy() {
        return stopTokenPolicy;
    }

    public int decoderStartTokenId() {
        return decoderStartTokenId;
    }

    public float temperature() {
        return temperature;
    }

    public int topK() {
        return topK;
    }

    private static int[] copyInputTokens(int[] inputTokenIds) {
        if (inputTokenIds == null || inputTokenIds.length == 0) {
            throw new IllegalArgumentException("inputTokenIds must not be empty");
        }
        int[] copy = Arrays.copyOf(inputTokenIds, inputTokenIds.length);
        for (int tokenId : copy) {
            if (tokenId < 0) {
                throw new IllegalArgumentException("inputTokenIds must not contain negative token ids");
            }
        }
        return copy;
    }

    private static int requirePositive(String name, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive: " + value);
        }
        return value;
    }
}
