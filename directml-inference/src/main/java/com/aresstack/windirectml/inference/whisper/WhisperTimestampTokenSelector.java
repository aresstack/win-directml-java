package com.aresstack.windirectml.inference.whisper;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Apply Whisper's token rules to decoder logits: suppressed non-speech tokens, the blank
 * suppression of the first token and the timestamp-pair and timestamp-probability rules
 * of the OpenAI reference decoder. Masking instead of choosing lets greedy decoding and
 * temperature sampling share exactly the same constraints.
 */
final class WhisperTimestampTokenSelector {
    private static final int MAX_INITIAL_TIMESTAMP_TICKS = 50;

    private final int firstTimestamp;
    private final int lastTimestamp;
    private final int textTokenCount;
    private final int endOfText;
    private final int[] suppressedTokens;
    private final int[] suppressedAtBegin;

    WhisperTimestampTokenSelector(int firstTimestamp, int lastTimestamp,
                                  int textTokenCount, int endOfText) {
        this(firstTimestamp, lastTimestamp, textTokenCount, endOfText, new int[0], new int[0]);
    }

    WhisperTimestampTokenSelector(int firstTimestamp, int lastTimestamp, int textTokenCount,
                                  int endOfText, int[] suppressedTokens, int[] suppressedAtBegin) {
        if (firstTimestamp <= textTokenCount || lastTimestamp < firstTimestamp
                || endOfText < textTokenCount || endOfText >= firstTimestamp) {
            throw new IllegalArgumentException("Invalid Whisper token ranges");
        }
        this.firstTimestamp = firstTimestamp;
        this.lastTimestamp = lastTimestamp;
        this.textTokenCount = textTokenCount;
        this.endOfText = endOfText;
        this.suppressedTokens = suppressedTokens.clone();
        this.suppressedAtBegin = suppressedAtBegin.clone();
    }

    /** Choose the most probable permitted token with timestamps enabled. */
    int select(float[] logits, List<Integer> emitted, int permittedLastTimestamp) {
        return argmax(constrain(logits, emitted, permittedLastTimestamp, true));
    }

    /**
     * Return a copy of the logits in which every forbidden token is negative infinity.
     * Without timestamps only text tokens and end-of-text remain permitted.
     */
    float[] constrain(float[] logits, List<Integer> emitted, int permittedLastTimestamp,
                      boolean timestamps) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(emitted, "emitted");
        if (logits.length <= lastTimestamp) {
            throw new IllegalArgumentException("Incomplete Whisper decoder logits");
        }
        float[] masked = new float[logits.length];
        Arrays.fill(masked, Float.NEGATIVE_INFINITY);
        System.arraycopy(logits, 0, masked, 0, textTokenCount);
        masked[endOfText] = logits[endOfText];
        int maxTimestamp = Math.min(lastTimestamp, permittedLastTimestamp);
        if (timestamps) {
            for (int token = firstTimestamp; token <= maxTimestamp; token++) {
                masked[token] = logits[token];
            }
        }
        for (int token : suppressedTokens) {
            if (token >= 0 && token < masked.length && token != endOfText) {
                masked[token] = Float.NEGATIVE_INFINITY;
            }
        }
        boolean initially = emitted.isEmpty();
        if (initially) {
            for (int token : suppressedAtBegin) {
                if (token >= 0 && token < masked.length) {
                    masked[token] = Float.NEGATIVE_INFINITY;
                }
            }
        }
        if (timestamps) {
            applyTimestampRules(masked, emitted, initially);
        }
        return masked;
    }

    private void applyTimestampRules(float[] masked, List<Integer> emitted, boolean initially) {
        boolean lastWasTimestamp = !initially && isTimestamp(emitted.get(emitted.size() - 1));
        boolean previousWasTimestamp = emitted.size() < 2
                || isTimestamp(emitted.get(emitted.size() - 2));
        if (lastWasTimestamp) {
            if (previousWasTimestamp) {
                fill(masked, firstTimestamp, lastTimestamp + 1);
            } else {
                fill(masked, 0, endOfText);
            }
        }
        int latestTimestamp = -1;
        for (int token : emitted) {
            if (isTimestamp(token)) {
                latestTimestamp = token;
            }
        }
        if (latestTimestamp >= 0) {
            boolean closing = lastWasTimestamp && !previousWasTimestamp;
            fill(masked, firstTimestamp, closing ? latestTimestamp : latestTimestamp + 1);
        }
        if (initially) {
            fill(masked, 0, firstTimestamp);
            fill(masked, firstTimestamp + MAX_INITIAL_TIMESTAMP_TICKS + 1, lastTimestamp + 1);
        }
        // Prefer a timestamp whenever all timestamps together outweigh every single other token.
        double timestampLogWeight = logSumExp(masked, firstTimestamp, lastTimestamp + 1);
        double bestOther = Float.NEGATIVE_INFINITY;
        for (int token = 0; token < firstTimestamp; token++) {
            bestOther = Math.max(bestOther, masked[token]);
        }
        if (timestampLogWeight > bestOther) {
            fill(masked, 0, firstTimestamp);
        }
    }

    static int argmax(float[] logits) {
        int best = 0;
        for (int token = 1; token < logits.length; token++) {
            if (logits[token] > logits[best]) {
                best = token;
            }
        }
        return best;
    }

    static double logSumExp(float[] values, int from, int to) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = from; i < to; i++) {
            max = Math.max(max, values[i]);
        }
        if (max == Double.NEGATIVE_INFINITY) {
            return max;
        }
        double sum = 0;
        for (int i = from; i < to; i++) {
            sum += Math.exp(values[i] - max);
        }
        return max + Math.log(sum);
    }

    private static void fill(float[] masked, int from, int to) {
        if (from < to) {
            Arrays.fill(masked, from, Math.min(to, masked.length), Float.NEGATIVE_INFINITY);
        }
    }

    boolean isTimestamp(int token) {
        return token >= firstTimestamp && token <= lastTimestamp;
    }

    int endOfText() {
        return endOfText;
    }
}
