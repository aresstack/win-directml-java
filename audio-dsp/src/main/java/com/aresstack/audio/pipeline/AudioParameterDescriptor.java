package com.aresstack.audio.pipeline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Describe one editable block parameter: its storage key, label, kind, default value and (for numbers)
 * range/step or (for choices) the allowed values. Single source used by the inspector to build the right
 * editor, by the registry to build default parameter maps, and by the validator to range-check.
 */
public final class AudioParameterDescriptor {

    private final String key;
    private final String label;
    private final AudioParameterType type;
    private final String defaultValue;
    private final double minimum;
    private final double maximum;
    private final double step;
    private final List<AudioParameterChoice> choices;

    private AudioParameterDescriptor(String key, String label, AudioParameterType type, String defaultValue,
                                     double minimum, double maximum, double step,
                                     List<AudioParameterChoice> choices) {
        this.key = key;
        this.label = label;
        this.type = type;
        this.defaultValue = defaultValue;
        this.minimum = minimum;
        this.maximum = maximum;
        this.step = step;
        this.choices = Collections.unmodifiableList(new ArrayList<AudioParameterChoice>(choices));
    }

    public static AudioParameterDescriptor integer(String key, String label, int defaultValue,
                                                   int minimum, int maximum) {
        return new AudioParameterDescriptor(key, label, AudioParameterType.INTEGER,
                String.valueOf(defaultValue), minimum, maximum, 1.0d,
                Collections.<AudioParameterChoice>emptyList());
    }

    public static AudioParameterDescriptor decimal(String key, String label, double defaultValue,
                                                   double minimum, double maximum, double step) {
        return new AudioParameterDescriptor(key, label, AudioParameterType.DECIMAL,
                formatNumber(defaultValue), minimum, maximum, step,
                Collections.<AudioParameterChoice>emptyList());
    }

    /** Render an integer-valued double without a trailing ".0" so defaults match the shipped strings. */
    private static String formatNumber(double value) {
        if (value == Math.rint(value) && !Double.isInfinite(value)) {
            return Long.toString((long) value);
        }
        return Double.toString(value);
    }

    public static AudioParameterDescriptor bool(String key, String label, boolean defaultValue) {
        return new AudioParameterDescriptor(key, label, AudioParameterType.BOOLEAN,
                String.valueOf(defaultValue), 0.0d, 0.0d, 0.0d,
                Collections.<AudioParameterChoice>emptyList());
    }

    /** A free-text parameter (for example a comma-separated list of channel weights or coordinates). */
    public static AudioParameterDescriptor text(String key, String label, String defaultValue) {
        return new AudioParameterDescriptor(key, label, AudioParameterType.TEXT,
                defaultValue == null ? "" : defaultValue, 0.0d, 0.0d, 0.0d,
                Collections.<AudioParameterChoice>emptyList());
    }

    public static AudioParameterDescriptor choice(String key, String label, String defaultValue,
                                                  List<AudioParameterChoice> choices) {
        return new AudioParameterDescriptor(key, label, AudioParameterType.CHOICE, defaultValue,
                0.0d, 0.0d, 0.0d, choices);
    }

    public String getKey() {
        return key;
    }

    public String getLabel() {
        return label;
    }

    public AudioParameterType getType() {
        return type;
    }

    public String getDefaultValue() {
        return defaultValue;
    }

    public double getMinimum() {
        return minimum;
    }

    public double getMaximum() {
        return maximum;
    }

    public double getStep() {
        return step;
    }

    public List<AudioParameterChoice> getChoices() {
        return choices;
    }
}
