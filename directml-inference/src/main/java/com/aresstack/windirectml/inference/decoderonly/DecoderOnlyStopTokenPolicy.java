package com.aresstack.windirectml.inference.decoderonly;

import java.util.Collection;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Decides whether a generated token should terminate decoder-only generation.
 */
@FunctionalInterface
public interface DecoderOnlyStopTokenPolicy {

    /**
     * Return true when generation should stop after the given token.
     */
    boolean shouldStop(int tokenId);

    /**
     * Create a policy backed by a defensive copy of the supplied token IDs.
     */
    static DecoderOnlyStopTokenPolicy fromTokenIds(Collection<Integer> tokenIds) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        Set<Integer> stopTokenIds = Set.copyOf(new HashSet<>(tokenIds));
        return stopTokenIds::contains;
    }
}
