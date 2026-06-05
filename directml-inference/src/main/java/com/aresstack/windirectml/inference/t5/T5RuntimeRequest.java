package com.aresstack.windirectml.inference.t5;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

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
    private final Set<Integer> suppressedTokenIds;
    private final int minimumTokensBeforeStop;

    public T5RuntimeRequest(int[] inputTokenIds,
                            int maxNewTokens,
                            T5StopTokenPolicy stopTokenPolicy,
                            int decoderStartTokenId,
                            float temperature,
                            int topK) {
        this(inputTokenIds, maxNewTokens, stopTokenPolicy, decoderStartTokenId, temperature, topK,
                Collections.<Integer>emptySet(), 0);
    }

    public T5RuntimeRequest(int[] inputTokenIds,
                            int maxNewTokens,
                            T5StopTokenPolicy stopTokenPolicy,
                            int decoderStartTokenId,
                            float temperature,
                            int topK,
                            Set<Integer> suppressedTokenIds,
                            int minimumTokensBeforeStop) {
        this.inputTokenIds = copyInputTokens(inputTokenIds);
        this.maxNewTokens = requirePositive("maxNewTokens", maxNewTokens);
        this.stopTokenPolicy = Objects.requireNonNull(stopTokenPolicy, "stopTokenPolicy");
        if (decoderStartTokenId < 0) {
            throw new IllegalArgumentException("decoderStartTokenId must not be negative: " + decoderStartTokenId);
        }
        if (minimumTokensBeforeStop < 0) {
            throw new IllegalArgumentException("minimumTokensBeforeStop must not be negative: " + minimumTokensBeforeStop);
        }
        this.decoderStartTokenId = decoderStartTokenId;
        this.temperature = temperature;
        this.topK = topK;
        this.suppressedTokenIds = sanitizeSuppressedTokenIds(suppressedTokenIds);
        this.minimumTokensBeforeStop = minimumTokensBeforeStop;
    }

    public static T5RuntimeRequest greedy(int[] inputTokenIds, int maxNewTokens, T5SpecialTokens specialTokens) {
        Objects.requireNonNull(specialTokens, "specialTokens");
        return new T5RuntimeRequest(inputTokenIds, maxNewTokens, specialTokens.stopAtEos(),
                specialTokens.decoderStartTokenId(), 0.0f, 0);
    }

    public static T5RuntimeRequest greedyText(int[] inputTokenIds, int maxNewTokens, T5SpecialTokens specialTokens) {
        Objects.requireNonNull(specialTokens, "specialTokens");
        LinkedHashSet<Integer> suppressed = new LinkedHashSet<Integer>();
        suppressed.add(specialTokens.padTokenId());
        suppressed.add(specialTokens.decoderStartTokenId());
        return new T5RuntimeRequest(inputTokenIds, maxNewTokens, specialTokens.stopAtEos(),
                specialTokens.decoderStartTokenId(), 0.0f, 0, suppressed, 1);
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

    public Set<Integer> suppressedTokenIds() {
        return suppressedTokenIds;
    }

    public int minimumTokensBeforeStop() {
        return minimumTokensBeforeStop;
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

    private static Set<Integer> sanitizeSuppressedTokenIds(Set<Integer> tokenIds) {
        if (tokenIds == null || tokenIds.isEmpty()) {
            return Collections.emptySet();
        }
        LinkedHashSet<Integer> copy = new LinkedHashSet<Integer>();
        for (Integer tokenId : tokenIds) {
            if (tokenId == null) {
                continue;
            }
            if (tokenId < 0) {
                throw new IllegalArgumentException("suppressedTokenIds must not contain negative token ids: " + tokenId);
            }
            copy.add(tokenId);
        }
        return Collections.unmodifiableSet(copy);
    }

    private static int requirePositive(String name, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive: " + value);
        }
        return value;
    }
}
