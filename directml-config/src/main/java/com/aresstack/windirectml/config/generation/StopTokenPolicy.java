package com.aresstack.windirectml.config.generation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Policy defining when text generation should stop.
 *
 * <p>Generation terminates when any of:
 * <ul>
 *   <li>{@code maxTokens} is reached, or</li>
 *   <li>a stop token / stop string is emitted.</li>
 * </ul>
 *
 * <p>Java-8 compatible.
 */
public final class StopTokenPolicy {

    /** Default: stop only on EOS or max-tokens. */
    private static final StopTokenPolicy EOS_ONLY =
            new StopTokenPolicy(Collections.<String>emptyList());

    private final List<String> stopStrings;

    private StopTokenPolicy(List<String> stopStrings) {
        this.stopStrings = Collections.unmodifiableList(new ArrayList<String>(stopStrings));
    }

    /** Default policy: stop on the model's EOS token or when maxTokens is reached. */
    public static StopTokenPolicy eosOnly() {
        return EOS_ONLY;
    }

    /**
     * Policy that additionally stops generation when any of the
     * given strings appear in the output.
     */
    public static StopTokenPolicy withStopStrings(String... stopStrings) {
        if (stopStrings == null || stopStrings.length == 0) {
            return EOS_ONLY;
        }
        List<String> validated = new ArrayList<String>(stopStrings.length);
        for (String stopString : stopStrings) {
            if (stopString == null || stopString.trim().isEmpty()) {
                throw new IllegalArgumentException("stop strings must not contain null or blank values");
            }
            validated.add(stopString);
        }
        return new StopTokenPolicy(validated);
    }

    /** Unmodifiable list of stop strings (may be empty). */
    public List<String> stopStrings() {
        return stopStrings;
    }

    @Override
    public String toString() {
        return stopStrings.isEmpty() ? "StopTokenPolicy[eos-only]"
                : "StopTokenPolicy[stopStrings=" + stopStrings + "]";
    }
}
