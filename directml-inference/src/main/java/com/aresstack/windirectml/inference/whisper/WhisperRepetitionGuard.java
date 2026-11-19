package com.aresstack.windirectml.inference.whisper;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Stop an unbounded greedy-decoding loop when a text-token sequence repeats.
 * Preserve the first occurrence and every previously completed timestamped segment.
 */
final class WhisperRepetitionGuard {
    private static final int MINIMUM_REPETITIONS = 5;
    private static final int MAXIMUM_PATTERN_TOKENS = 32;

    private final int textTokenCount;
    private final int firstTimestamp;
    private final int lastTimestamp;
    private final Function<List<Integer>, String> decodeText;

    WhisperRepetitionGuard(int textTokenCount, int firstTimestamp, int lastTimestamp,
                           Function<List<Integer>, String> decodeText) {
        this.textTokenCount = textTokenCount;
        this.firstTimestamp = firstTimestamp;
        this.lastTimestamp = lastTimestamp;
        this.decodeText = Objects.requireNonNull(decodeText, "decodeText");
    }

    boolean stopAndTrimRepeatedText(List<Integer> tokens) {
        int start = startOfCurrentText(tokens);
        int length = tokens.size() - start;
        int maximumWidth = Math.min(MAXIMUM_PATTERN_TOKENS, length / MINIMUM_REPETITIONS);
        for (int width = 1; width <= maximumWidth; width++) {
            if (!hasRepeatedSuffix(tokens, width)) {
                continue;
            }
            int earliestCopy = tokens.size() - MINIMUM_REPETITIONS * width;
            while (earliestCopy - width >= start
                    && equalBlocks(tokens, earliestCopy - width, earliestCopy, width)) {
                earliestCopy -= width;
            }
            int retainedEnd = hasEquivalentPrefix(tokens, start, earliestCopy, width)
                    ? earliestCopy : earliestCopy + width;
            tokens.subList(retainedEnd, tokens.size()).clear();
            return true;
        }
        return false;
    }

    private boolean hasEquivalentPrefix(List<Integer> tokens, int start, int firstCopy, int width) {
        String repeatedText = decodeText.apply(tokens.subList(firstCopy, firstCopy + width)).strip();
        if (repeatedText.isEmpty()) {
            return false;
        }
        int maximumLength = Math.min(MAXIMUM_PATTERN_TOKENS + 4, firstCopy - start);
        for (int length = 1; length <= maximumLength; length++) {
            String precedingText = decodeText.apply(tokens.subList(firstCopy - length, firstCopy)).strip();
            if (repeatedText.equals(precedingText)) {
                return true;
            }
        }
        return false;
    }

    private int startOfCurrentText(List<Integer> tokens) {
        for (int index = tokens.size() - 1; index >= 0; index--) {
            int token = tokens.get(index);
            if (token >= firstTimestamp && token <= lastTimestamp) {
                return index + 1;
            }
            if (token < 0 || token >= textTokenCount) {
                return tokens.size();
            }
        }
        return 0;
    }

    private static boolean hasRepeatedSuffix(List<Integer> tokens, int width) {
        int end = tokens.size();
        for (int copy = 1; copy < MINIMUM_REPETITIONS; copy++) {
            if (!equalBlocks(tokens, end - width, end - (copy + 1) * width, width)) {
                return false;
            }
        }
        return true;
    }

    private static boolean equalBlocks(List<Integer> tokens, int first, int second, int length) {
        for (int index = 0; index < length; index++) {
            if (!tokens.get(first + index).equals(tokens.get(second + index))) {
                return false;
            }
        }
        return true;
    }
}
