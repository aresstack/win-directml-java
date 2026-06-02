package com.aresstack.windirectml.config.generation;

/**
 * Configuration for token sampling during text generation.
 *
 * <p>V1 supports greedy decoding only (temperature=0, top-k=1).
 * Future versions may add nucleus (top-p), top-k, repetition-penalty, etc.
 *
 * <p>Java-8 compatible (no records).
 */
public final class SamplerConfig {

    /**
     * Singleton greedy sampler (temperature=0, topK=1).
     */
    private static final SamplerConfig GREEDY = new SamplerConfig(0.0f, 1);

    private final float temperature;
    private final int topK;

    private SamplerConfig(float temperature, int topK) {
        this.temperature = temperature;
        this.topK = topK;
    }

    /**
     * Returns the greedy decoding configuration (argmax).
     */
    public static SamplerConfig greedy() {
        return GREEDY;
    }

    /**
     * Returns a temperature-based sampling configuration.
     *
     * @param temperature temperature value; 0 = greedy, higher = more random.
     * @param topK        number of top candidates to consider; 1 = greedy.
     * @throws IllegalArgumentException if temperature is negative or topK &lt; 1.
     */
    public static SamplerConfig of(float temperature, int topK) {
        if (Float.isNaN(temperature) || Float.isInfinite(temperature)) {
            throw new IllegalArgumentException("temperature must be finite");
        }
        if (temperature < 0) {
            throw new IllegalArgumentException("temperature must be >= 0");
        }
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be >= 1");
        }
        if (temperature == 0.0f && topK == 1) {
            return GREEDY;
        }
        return new SamplerConfig(temperature, topK);
    }

    public float temperature() {
        return temperature;
    }

    public int topK() {
        return topK;
    }

    public boolean isGreedy() {
        return temperature == 0.0f && topK == 1;
    }

    @Override
    public String toString() {
        return isGreedy() ? "SamplerConfig[greedy]"
                : "SamplerConfig[temperature=" + temperature + ", topK=" + topK + "]";
    }
}
