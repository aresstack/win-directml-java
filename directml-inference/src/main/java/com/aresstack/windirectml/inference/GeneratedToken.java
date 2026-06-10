package com.aresstack.windirectml.inference;

/**
 * Token event emitted while a text-generation model is producing output.
 */
public final class GeneratedToken {

    private final int tokenId;
    private final String textSoFar;
    private final String delta;

    public GeneratedToken(int tokenId, String textSoFar, String delta) {
        this.tokenId = tokenId;
        this.textSoFar = textSoFar == null ? "" : textSoFar;
        this.delta = delta == null ? "" : delta;
    }

    public static GeneratedToken bufferedText(String text) {
        String safeText = text == null ? "" : text;
        return new GeneratedToken(-1, safeText, safeText);
    }

    public int tokenId() {
        return tokenId;
    }

    public String textSoFar() {
        return textSoFar;
    }

    public String delta() {
        return delta;
    }
}
