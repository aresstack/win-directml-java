package com.aresstack.windirectml.inference.smollm2;

/**
 * Special token ids declared by the SmolLM2 config.
 */
public record SmolLM2SpecialTokens(int bosTokenId, int eosTokenId, Integer padTokenId) {
    public boolean isEos(int tokenId) {
        return tokenId == eosTokenId;
    }
}
