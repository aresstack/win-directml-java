package com.aresstack.windirectml.inference.whisper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Hold recognized text together with intervals confirmed by Whisper timestamp tokens. */
public final class TranscriptionResult {
    private final String text;
    private final String unalignedText;
    private final List<TranscriptionSegment> segments;

    public TranscriptionResult(String text, String unalignedText, List<TranscriptionSegment> segments) {
        this.text = Objects.requireNonNull(text, "text").strip();
        this.unalignedText = Objects.requireNonNull(unalignedText, "unalignedText").strip();
        this.segments = Collections.unmodifiableList(new ArrayList<>(
                Objects.requireNonNull(segments, "segments")));
    }

    public String getText() {
        return text;
    }

    public String getUnalignedText() {
        return unalignedText;
    }

    public List<TranscriptionSegment> getSegments() {
        return segments;
    }

    public boolean hasUnalignedText() {
        return !unalignedText.isEmpty();
    }
}
