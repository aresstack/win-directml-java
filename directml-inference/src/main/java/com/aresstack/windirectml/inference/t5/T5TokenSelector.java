package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Selects the next token from decoder logits.
 */
public interface T5TokenSelector {
    int select(float[] logits);

    static T5TokenSelector greedy() {
        return GreedyT5TokenSelector.INSTANCE;
    }
}

final class GreedyT5TokenSelector implements T5TokenSelector {
    static final GreedyT5TokenSelector INSTANCE = new GreedyT5TokenSelector();

    private GreedyT5TokenSelector() {
    }

    @Override
    public int select(float[] logits) {
        Objects.requireNonNull(logits, "logits");
        if (logits.length == 0) {
            throw new IllegalArgumentException("logits must not be empty");
        }
        int bestToken = 0;
        float bestLogit = sanitize(logits[0]);
        for (int token = 1; token < logits.length; token++) {
            float logit = sanitize(logits[token]);
            if (logit > bestLogit) {
                bestLogit = logit;
                bestToken = token;
            }
        }
        return bestToken;
    }

    private static float sanitize(float value) {
        if (Float.isNaN(value)) {
            return -Float.MAX_VALUE;
        }
        if (value == Float.POSITIVE_INFINITY) {
            return Float.MAX_VALUE;
        }
        if (value == Float.NEGATIVE_INFINITY) {
            return -Float.MAX_VALUE;
        }
        return value;
    }
}
