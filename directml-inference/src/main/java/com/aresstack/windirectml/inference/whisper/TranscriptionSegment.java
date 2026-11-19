package com.aresstack.windirectml.inference.whisper;

import java.util.Objects;

/** Describe a Whisper-recognized text interval relative to the beginning of the recording. */
public final class TranscriptionSegment {
    private final long startMillis;
    private final long endMillis;
    private final String text;

    public TranscriptionSegment(long startMillis, long endMillis, String text) {
        if (startMillis < 0 || endMillis <= startMillis) {
            throw new IllegalArgumentException("A transcription segment requires a positive time interval");
        }
        this.text = Objects.requireNonNull(text, "text").strip();
        if (this.text.isEmpty()) {
            throw new IllegalArgumentException("A transcription segment requires text");
        }
        this.startMillis = startMillis;
        this.endMillis = endMillis;
    }

    public long getStartMillis() {
        return startMillis;
    }

    public long getEndMillis() {
        return endMillis;
    }

    public String getText() {
        return text;
    }
}
