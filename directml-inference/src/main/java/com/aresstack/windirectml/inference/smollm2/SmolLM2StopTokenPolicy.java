package com.aresstack.windirectml.inference.smollm2;

/**
 * EOS-based stop policy for SmolLM2 generation.
 */
public final class SmolLM2StopTokenPolicy {

    private final int eosTokenId;

    public SmolLM2StopTokenPolicy(int eosTokenId) {
        this.eosTokenId = eosTokenId;
    }

    public boolean shouldStop(int tokenId) {
        return tokenId == eosTokenId;
    }
}
