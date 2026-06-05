package com.aresstack.windirectml.inference.t5;

/**
 * Placeholder state boundary for the future T5 decoder cache.
 */
public final class T5DecoderState {
    private final int generatedTokens;

    public T5DecoderState(int generatedTokens) {
        if (generatedTokens < 0) {
            throw new IllegalArgumentException("generatedTokens must not be negative: " + generatedTokens);
        }
        this.generatedTokens = generatedTokens;
    }

    public int generatedTokens() {
        return generatedTokens;
    }
}
