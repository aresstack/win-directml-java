package com.aresstack.windirectml.inference.t5;

import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Token-based stopping policy for the T5 decoder loop.
 */
public final class T5StopTokenPolicy {
    private final Set<Integer> stopTokenIds;

    private T5StopTokenPolicy(Set<Integer> stopTokenIds) {
        this.stopTokenIds = Collections.unmodifiableSet(new HashSet<>(Objects.requireNonNull(stopTokenIds, "stopTokenIds")));
    }

    public static T5StopTokenPolicy stopAtEos(int eosTokenId) {
        if (eosTokenId < 0) {
            throw new IllegalArgumentException("eosTokenId must not be negative: " + eosTokenId);
        }
        Set<Integer> tokens = new HashSet<>();
        tokens.add(eosTokenId);
        return new T5StopTokenPolicy(tokens);
    }

    public static T5StopTokenPolicy stopAtAny(Set<Integer> tokenIds) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        Set<Integer> copy = new HashSet<>();
        for (Integer tokenId : tokenIds) {
            if (tokenId == null || tokenId < 0) {
                throw new IllegalArgumentException("stop token ids must not contain null or negative values");
            }
            copy.add(tokenId);
        }
        return new T5StopTokenPolicy(copy);
    }

    public static T5StopTokenPolicy neverStop() {
        return new T5StopTokenPolicy(Collections.<Integer>emptySet());
    }

    public boolean shouldStop(int tokenId) {
        return stopTokenIds.contains(tokenId);
    }

    public Set<Integer> stopTokenIds() {
        return stopTokenIds;
    }
}
