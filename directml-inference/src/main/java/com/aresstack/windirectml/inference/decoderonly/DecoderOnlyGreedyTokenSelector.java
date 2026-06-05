package com.aresstack.windirectml.inference.decoderonly;

import java.util.Objects;

/**
 * Selects the next decoder-only token using greedy decoding.
 */
public final class DecoderOnlyGreedyTokenSelector {

    private final DecoderOnlyStopTokenPolicy stopTokenPolicy;

    public DecoderOnlyGreedyTokenSelector(DecoderOnlyStopTokenPolicy stopTokenPolicy) {
        this.stopTokenPolicy = Objects.requireNonNull(stopTokenPolicy, "stopTokenPolicy");
    }

    /**
     * Apply repetition penalty in-place and return the highest-logit token.
     */
    public int selectNextToken(float[] logits, DecoderOnlyGeneratedTokens generatedTokens,
                               float repetitionPenalty) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(generatedTokens, "generatedTokens");
        DecoderOnlyMath.applyRepetitionPenalty(
                logits,
                generatedTokens.backingArray(),
                generatedTokens.count(),
                repetitionPenalty);
        return DecoderOnlyMath.argmax(logits);
    }

    public boolean shouldStop(int tokenId) {
        return stopTokenPolicy.shouldStop(tokenId);
    }
}
