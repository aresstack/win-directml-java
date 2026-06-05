package com.aresstack.windirectml.inference.t5;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Special token ids required by the T5 generation boundary.
 */
public final class T5SpecialTokens {
    private final int padTokenId;
    private final int eosTokenId;
    private final int decoderStartTokenId;

    public T5SpecialTokens(int padTokenId, int eosTokenId, int decoderStartTokenId) {
        this.padTokenId = padTokenId;
        this.eosTokenId = eosTokenId;
        this.decoderStartTokenId = decoderStartTokenId;
        validate();
    }

    public static T5SpecialTokens from(T5Config config) {
        Objects.requireNonNull(config, "config");
        return new T5SpecialTokens(config.padTokenId(), config.eosTokenId(), config.decoderStartTokenId());
    }

    public int padTokenId() {
        return padTokenId;
    }

    public int eosTokenId() {
        return eosTokenId;
    }

    public int decoderStartTokenId() {
        return decoderStartTokenId;
    }

    public void validate() {
        requireNonNegative("padTokenId", padTokenId);
        requireNonNegative("eosTokenId", eosTokenId);
        requireNonNegative("decoderStartTokenId", decoderStartTokenId);
    }

    public T5StopTokenPolicy stopAtEos() {
        return T5StopTokenPolicy.stopAtEos(eosTokenId);
    }

    public T5StopTokenPolicy stopAtAny(Set<Integer> tokenIds) {
        return T5StopTokenPolicy.stopAtAny(tokenIds);
    }

    public Map<String, Object> toManifest() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("padTokenId", padTokenId);
        out.put("eosTokenId", eosTokenId);
        out.put("decoderStartTokenId", decoderStartTokenId);
        return out;
    }

    private static void requireNonNegative(String name, int value) {
        if (value < 0) {
            throw new IllegalArgumentException("T5 special token " + name + " must not be negative: " + value);
        }
    }
}
