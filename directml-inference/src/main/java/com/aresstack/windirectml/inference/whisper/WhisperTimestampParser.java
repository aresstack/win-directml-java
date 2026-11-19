package com.aresstack.windirectml.inference.whisper;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/** Recover speech intervals only from timestamp tokens emitted by Whisper. */
final class WhisperTimestampParser {
    private static final long MILLISECONDS_PER_TOKEN = 20;

    private final Function<List<Integer>, String> decodeText;
    private final int textTokenCount;
    private final int firstTimestamp;
    private final int lastTimestamp;

    WhisperTimestampParser(WhisperTokenizer tokenizer) {
        this(tokenizer.requireSpecial("<|0.00|>"),
                tokenizer.requireSpecial("<|30.00|>"),
                tokenizer.textTokenCount(), tokenizer::decode);
    }

    WhisperTimestampParser(int firstTimestamp, int lastTimestamp, int textTokenCount,
                           Function<List<Integer>, String> decodeText) {
        if (firstTimestamp < 0 || lastTimestamp < firstTimestamp || textTokenCount < 1) {
            throw new IllegalArgumentException("Invalid Whisper token boundaries");
        }
        this.firstTimestamp = firstTimestamp;
        this.lastTimestamp = lastTimestamp;
        this.textTokenCount = textTokenCount;
        this.decodeText = Objects.requireNonNull(decodeText, "decodeText");
    }

    int firstTimestamp() {
        return firstTimestamp;
    }

    int lastTimestamp() {
        return lastTimestamp;
    }

    TranscriptionResult parse(List<Integer> tokens, long windowStartMillis, long windowDurationMillis) {
        Objects.requireNonNull(tokens, "tokens");
        if (windowStartMillis < 0 || windowDurationMillis < 0) {
            throw new IllegalArgumentException("Invalid Whisper window interval");
        }
        List<Integer> allTextTokens = new ArrayList<>();
        List<Integer> remainingTextTokens = new ArrayList<>();
        List<Integer> unalignedTokens = new ArrayList<>();
        List<TranscriptionSegment> segments = new ArrayList<>();
        Long segmentStart = null;

        for (Integer token : tokens) {
            if (token != null && token >= firstTimestamp && token <= lastTimestamp) {
                long positionMillis = Math.min(windowDurationMillis,
                        (long) (token - firstTimestamp) * MILLISECONDS_PER_TOKEN);
                if (segmentStart == null) {
                    if (!remainingTextTokens.isEmpty()) {
                        unalignedTokens.addAll(remainingTextTokens);
                        remainingTextTokens.clear();
                    }
                    segmentStart = positionMillis;
                } else if (remainingTextTokens.isEmpty()) {
                    segmentStart = positionMillis;
                } else {
                    String segmentText = decodeText.apply(remainingTextTokens).strip();
                    if (positionMillis > segmentStart && !segmentText.isEmpty()) {
                        segments.add(new TranscriptionSegment(windowStartMillis + segmentStart,
                                windowStartMillis + positionMillis, segmentText));
                    } else {
                        unalignedTokens.addAll(remainingTextTokens);
                    }
                    remainingTextTokens.clear();
                    segmentStart = null;
                }
            } else if (token != null && token >= 0 && token < textTokenCount) {
                allTextTokens.add(token);
                remainingTextTokens.add(token);
            }
        }
        unalignedTokens.addAll(remainingTextTokens);

        return new TranscriptionResult(decodeText.apply(allTextTokens),
                decodeText.apply(unalignedTokens), segments);
    }
}
