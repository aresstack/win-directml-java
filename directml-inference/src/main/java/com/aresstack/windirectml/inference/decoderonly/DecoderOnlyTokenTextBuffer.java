package com.aresstack.windirectml.inference.decoderonly;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Builds generated text incrementally from decoded token fragments.
 */
public final class DecoderOnlyTokenTextBuffer {

    private final ByteArrayOutputStream decodedBytes;
    private String previousText;

    public DecoderOnlyTokenTextBuffer(int initialCapacity) {
        if (initialCapacity < 0) {
            throw new IllegalArgumentException("initialCapacity must not be negative");
        }
        this.decodedBytes = new ByteArrayOutputStream(initialCapacity);
        this.previousText = "";
    }

    /**
     * Append one decoded token fragment and return the full text plus delta.
     */
    public DecoderOnlyTokenText appendDecodedTokenText(String tokenText) {
        Objects.requireNonNull(tokenText, "tokenText");
        byte[] tokenBytes = tokenText.getBytes(StandardCharsets.UTF_8);
        decodedBytes.write(tokenBytes, 0, tokenBytes.length);
        String fullText = decodedBytes.toString(StandardCharsets.UTF_8);
        String delta = fullText.length() >= previousText.length()
                ? fullText.substring(previousText.length())
                : "";
        previousText = fullText;
        return new DecoderOnlyTokenText(fullText, delta);
    }

    public String fullText() {
        return previousText;
    }

    public record DecoderOnlyTokenText(String fullText, String delta) {
    }
}
