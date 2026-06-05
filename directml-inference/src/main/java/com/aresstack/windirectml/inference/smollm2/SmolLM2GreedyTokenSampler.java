package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGreedyTokenSelector;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;

import java.util.Objects;

/**
 * Selects the highest-logit token.
 */
final class SmolLM2GreedyTokenSampler implements SmolLM2TokenSampler {

    private final DecoderOnlyGreedyTokenSelector tokenSelector;
    private final float repetitionPenalty;

    SmolLM2GreedyTokenSampler(DecoderOnlyStopTokenPolicy stopTokenPolicy, double repetitionPenalty) {
        this.tokenSelector = new DecoderOnlyGreedyTokenSelector(Objects.requireNonNull(stopTokenPolicy, "stopTokenPolicy"));
        this.repetitionPenalty = (float) repetitionPenalty;
    }

    @Override
    public int selectNextToken(float[] logits, DecoderOnlyGeneratedTokens generatedTokens) {
        return tokenSelector.selectNextToken(logits, generatedTokens, repetitionPenalty);
    }

    @Override
    public boolean shouldStop(int tokenId) {
        return tokenSelector.shouldStop(tokenId);
    }
}
