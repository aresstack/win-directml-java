package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;

/**
 * Selects the next SmolLM2 token from logits.
 */
interface SmolLM2TokenSampler {
    int selectNextToken(float[] logits, DecoderOnlyGeneratedTokens generatedTokens);

    boolean shouldStop(int tokenId);
}
