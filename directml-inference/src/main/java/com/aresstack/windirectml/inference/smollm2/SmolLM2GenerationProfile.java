package com.aresstack.windirectml.inference.smollm2;

import java.util.List;

/**
 * Timing profile for one SmolLM2 generation run.
 */
public record SmolLM2GenerationProfile(long runtimeNanos,
                                       long tokenizeNanos,
                                       long prefillNanos,
                                       long decoderStepNanos,
                                       long lmHeadNanos,
                                       long tokenSelectNanos,
                                       long detokenizeNanos,
                                       SmolLM2ReferenceHotspotProfile referenceHotspots,
                                       List<String> stepTopK) {

    public SmolLM2GenerationProfile {
        runtimeNanos = nonNegative(runtimeNanos);
        tokenizeNanos = nonNegative(tokenizeNanos);
        prefillNanos = nonNegative(prefillNanos);
        decoderStepNanos = nonNegative(decoderStepNanos);
        lmHeadNanos = nonNegative(lmHeadNanos);
        tokenSelectNanos = nonNegative(tokenSelectNanos);
        detokenizeNanos = nonNegative(detokenizeNanos);
        referenceHotspots = referenceHotspots == null
                ? SmolLM2ReferenceHotspotProfile.empty()
                : referenceHotspots;
        stepTopK = stepTopK == null ? List.of() : List.copyOf(stepTopK);
    }

    public SmolLM2GenerationProfile(long runtimeNanos,
                                    long tokenizeNanos,
                                    long prefillNanos,
                                    long decoderStepNanos,
                                    long lmHeadNanos,
                                    long tokenSelectNanos,
                                    long detokenizeNanos,
                                    SmolLM2ReferenceHotspotProfile referenceHotspots) {
        this(runtimeNanos, tokenizeNanos, prefillNanos, decoderStepNanos, lmHeadNanos,
                tokenSelectNanos, detokenizeNanos, referenceHotspots, List.of());
    }

    public SmolLM2GenerationProfile(long runtimeNanos,
                                    long tokenizeNanos,
                                    long prefillNanos,
                                    long decoderStepNanos,
                                    long lmHeadNanos,
                                    long tokenSelectNanos,
                                    long detokenizeNanos) {
        this(runtimeNanos, tokenizeNanos, prefillNanos, decoderStepNanos, lmHeadNanos,
                tokenSelectNanos, detokenizeNanos, SmolLM2ReferenceHotspotProfile.empty(), List.of());
    }

    public static SmolLM2GenerationProfile empty() {
        return new SmolLM2GenerationProfile(0L, 0L, 0L, 0L, 0L, 0L, 0L,
                SmolLM2ReferenceHotspotProfile.empty(), List.of());
    }

    public SmolLM2GenerationProfile withTextTimings(long newRuntimeNanos,
                                                     long newTokenizeNanos,
                                                     long newDetokenizeNanos) {
        return new SmolLM2GenerationProfile(
                newRuntimeNanos,
                newTokenizeNanos,
                prefillNanos,
                decoderStepNanos,
                lmHeadNanos,
                tokenSelectNanos,
                newDetokenizeNanos,
                referenceHotspots,
                stepTopK);
    }

    public long runtimeMillis() {
        return toMillis(runtimeNanos);
    }

    public long tokenizeMillis() {
        return toMillis(tokenizeNanos);
    }

    public long prefillMillis() {
        return toMillis(prefillNanos);
    }

    public long decoderStepMillis() {
        return toMillis(decoderStepNanos);
    }

    public long lmHeadMillis() {
        return toMillis(lmHeadNanos);
    }

    public long tokenSelectMillis() {
        return toMillis(tokenSelectNanos);
    }

    public long detokenizeMillis() {
        return toMillis(detokenizeNanos);
    }

    public long averageTokenRuntimeMillis(int generatedTokenCount) {
        if (generatedTokenCount <= 0) {
            return 0L;
        }
        return toMillis(runtimeNanos / generatedTokenCount);
    }

    private static long nonNegative(long value) {
        return Math.max(0L, value);
    }

    private static long toMillis(long nanos) {
        return nanos / 1_000_000L;
    }
}
