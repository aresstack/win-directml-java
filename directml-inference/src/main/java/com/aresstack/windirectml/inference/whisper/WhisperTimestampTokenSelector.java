package com.aresstack.windirectml.inference.whisper;

import java.util.List;
import java.util.Objects;

/**
 * Apply Whisper's timestamp-pair and timestamp-probability rules to greedy decoding.
 * Keep the token policy independent of the inference session.
 */
final class WhisperTimestampTokenSelector {
    private static final int MAX_INITIAL_TIMESTAMP_TICKS = 50;

    private final int firstTimestamp;
    private final int lastTimestamp;
    private final int textTokenCount;
    private final int endOfText;

    WhisperTimestampTokenSelector(int firstTimestamp, int lastTimestamp,
                                  int textTokenCount, int endOfText) {
        if (firstTimestamp <= textTokenCount || lastTimestamp < firstTimestamp
                || endOfText < textTokenCount || endOfText >= firstTimestamp) {
            throw new IllegalArgumentException("Invalid Whisper token ranges");
        }
        this.firstTimestamp = firstTimestamp;
        this.lastTimestamp = lastTimestamp;
        this.textTokenCount = textTokenCount;
        this.endOfText = endOfText;
    }

    int select(float[] logits, List<Integer> emitted, int permittedLastTimestamp) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(emitted, "emitted");
        if (logits.length <= lastTimestamp) {
            throw new IllegalArgumentException("Incomplete Whisper decoder logits");
        }

        boolean initially = emitted.isEmpty();
        boolean lastWasTimestamp = !initially && isTimestamp(emitted.get(emitted.size() - 1));
        boolean previousWasTimestamp = emitted.size() < 2
                || isTimestamp(emitted.get(emitted.size() - 2));
        boolean mustCloseSegment = lastWasTimestamp && !previousWasTimestamp;
        boolean mustGenerateText = lastWasTimestamp && previousWasTimestamp;

        int minTimestamp = firstTimestamp;
        for (int token : emitted) {
            if (isTimestamp(token)) {
                minTimestamp = Math.max(minTimestamp, token + (mustCloseSegment ? 0 : 1));
            }
        }
        int maxTimestamp = Math.min(lastTimestamp, permittedLastTimestamp);
        if (initially) {
            maxTimestamp = Math.min(maxTimestamp, firstTimestamp + MAX_INITIAL_TIMESTAMP_TICKS);
        }

        int bestText = endOfText;
        float bestTextScore = initially || mustCloseSegment
                ? Float.NEGATIVE_INFINITY : logits[endOfText];
        if (!initially && !mustCloseSegment) {
            for (int token = 0; token < textTokenCount; token++) {
                if (logits[token] > bestTextScore) {
                    bestTextScore = logits[token];
                    bestText = token;
                }
            }
        }
        if (mustCloseSegment) {
            bestTextScore = logits[endOfText];
            bestText = endOfText;
        }

        int bestTimestamp = -1;
        float bestTimestampScore = Float.NEGATIVE_INFINITY;
        double totalTimestampWeight = 0;
        if (!mustGenerateText) {
            for (int token = minTimestamp; token <= maxTimestamp; token++) {
                float score = logits[token];
                if (score > bestTimestampScore) {
                    bestTimestampScore = score;
                    bestTimestamp = token;
                }
            }
            if (bestTimestamp >= 0 && Float.isFinite(bestTimestampScore)) {
                for (int token = minTimestamp; token <= maxTimestamp; token++) {
                    totalTimestampWeight += Math.exp(logits[token] - bestTimestampScore);
                }
            }
        }

        if (bestTimestamp >= 0) {
            double timestampLogWeight = bestTimestampScore + Math.log(totalTimestampWeight);
            if (initially || mustCloseSegment || timestampLogWeight > bestTextScore) {
                if (mustCloseSegment && logits[endOfText] > timestampLogWeight) {
                    return endOfText;
                }
                return bestTimestamp;
            }
        }
        return bestText;
    }

    private boolean isTimestamp(int token) {
        return token >= firstTimestamp && token <= lastTimestamp;
    }
}
