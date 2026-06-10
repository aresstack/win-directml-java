package com.aresstack.windirectml.inference;

/**
 * Collects streamed token deltas into a final text buffer.
 */
public final class BufferingGenerationTokenSink implements GenerationTokenSink {

    private final StringBuilder text = new StringBuilder();
    private int tokenCount;

    @Override
    public void onToken(GeneratedToken token) {
        if (token == null) {
            return;
        }
        text.append(token.delta());
        tokenCount++;
    }

    public String text() {
        return text.toString();
    }

    public int tokenCount() {
        return tokenCount;
    }
}
