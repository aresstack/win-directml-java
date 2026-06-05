package com.aresstack.windirectml.inference.smollm2;

/**
 * Optional generation parameters for the SmolLM2 runtime.
 */
public record SmolLM2GenerationOptions(Double temperature,
                                       Integer topK,
                                       Double topP,
                                       Long randomSeed,
                                       Double repetitionPenalty) {
    private static final double GREEDY_TEMPERATURE = 0.0d;
    private static final double DEFAULT_REPETITION_PENALTY = 1.0d;

    public SmolLM2GenerationOptions(Double temperature, Integer topK, Double topP) {
        this(temperature, topK, topP, null, DEFAULT_REPETITION_PENALTY);
    }

    public SmolLM2GenerationOptions {
        if (temperature != null && temperature < 0.0d) {
            throw new IllegalArgumentException("temperature must be >= 0");
        }
        if (topK != null && topK < 0) {
            throw new IllegalArgumentException("topK must be >= 0");
        }
        if (topP != null && (topP <= 0.0d || topP > 1.0d)) {
            throw new IllegalArgumentException("topP must be > 0 and <= 1");
        }
        if (repetitionPenalty != null && repetitionPenalty <= 0.0d) {
            throw new IllegalArgumentException("repetitionPenalty must be > 0");
        }
    }

    public static SmolLM2GenerationOptions greedy() {
        return new SmolLM2GenerationOptions(
                GREEDY_TEMPERATURE,
                null,
                null,
                null,
                DEFAULT_REPETITION_PENALTY);
    }

    public static SmolLM2GenerationOptions sampling(double temperature, Integer topK, Double topP, long randomSeed) {
        return new SmolLM2GenerationOptions(temperature, topK, topP, randomSeed, DEFAULT_REPETITION_PENALTY);
    }

    public boolean usesSampling() {
        return temperature != null && temperature > 0.0d;
    }

    public double effectiveTemperature() {
        return temperature == null ? GREEDY_TEMPERATURE : temperature;
    }

    public int effectiveTopK() {
        return topK == null ? 0 : topK;
    }

    public double effectiveTopP() {
        return topP == null ? 1.0d : topP;
    }

    public double effectiveRepetitionPenalty() {
        return repetitionPenalty == null ? DEFAULT_REPETITION_PENALTY : repetitionPenalty;
    }
}
