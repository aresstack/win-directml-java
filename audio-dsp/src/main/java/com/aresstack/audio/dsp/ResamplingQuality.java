package com.aresstack.audio.dsp;

/** Select the interpolation cost used by the configurable resampler block. */
public enum ResamplingQuality {
    FAST,
    BALANCED,
    HIGH;

    public static ResamplingQuality parse(String value) {
        if (value == null) {
            return HIGH;
        }
        try {
            return ResamplingQuality.valueOf(value.trim().toUpperCase());
        } catch (IllegalArgumentException ex) {
            return HIGH;
        }
    }
}
