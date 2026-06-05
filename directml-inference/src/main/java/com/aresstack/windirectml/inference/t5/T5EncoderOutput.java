package com.aresstack.windirectml.inference.t5;

/**
 * Placeholder handle boundary for future T5 encoder outputs.
 */
public final class T5EncoderOutput {
    private final int inputTokens;

    public T5EncoderOutput(int inputTokens) {
        if (inputTokens <= 0) {
            throw new IllegalArgumentException("inputTokens must be positive: " + inputTokens);
        }
        this.inputTokens = inputTokens;
    }

    public int inputTokens() {
        return inputTokens;
    }
}
