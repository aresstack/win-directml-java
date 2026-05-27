package com.aresstack.windirectml.config.generation;

/**
 * Type-safe identifier for a text-generation model checkpoint.
 *
 * <p>Wraps the HuggingFace-style {@code org/model-name} string
 * used throughout this project's registries (e.g.
 * {@code "microsoft/Phi-3-mini-4k-instruct-onnx"}).
 *
 * <p>Java-8 compatible (no records).
 */
public final class GenerationModelId {

    private final String value;

    private GenerationModelId(String value) {
        this.value = value;
    }

    /**
     * Creates a model ID from a non-null, non-blank string.
     *
     * @throws IllegalArgumentException if {@code value} is null or blank.
     */
    public static GenerationModelId of(String value) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException("model ID must not be blank");
        }
        return new GenerationModelId(value.trim());
    }

    /** The raw model-id string (e.g. {@code "microsoft/Phi-3-mini-4k-instruct-onnx"}). */
    public String value() {
        return value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof GenerationModelId)) return false;
        GenerationModelId that = (GenerationModelId) o;
        return value.equals(that.value);
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }

    @Override
    public String toString() {
        return value;
    }
}
