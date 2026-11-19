package com.aresstack.audio.pipeline;

/** One selectable value of a CHOICE parameter: the stored value plus a human-readable label. */
public final class AudioParameterChoice {

    private final String value;
    private final String label;

    public AudioParameterChoice(String value, String label) {
        this.value = value == null ? "" : value;
        this.label = label == null || label.trim().isEmpty() ? this.value : label;
    }

    public String getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    @Override
    public String toString() {
        return label;
    }
}
