package com.aresstack.audio.pipeline;

/** Groups blocks for the "Add block" dialog and future editor sections (see requirements section 12.2). */
public enum AudioBlockCategory {
    GENERAL("General"),
    INPUT_CHANNEL("Input / Channel"),
    FILTERS_EQ("Filters / Equalization"),
    DYNAMICS("Dynamics"),
    NOISE_REDUCTION("Noise Reduction"),
    SPEECH_ENHANCEMENT("Speech Enhancement"),
    DEREVERBERATION("Dereverberation"),
    ANALYSIS("Analysis"),
    OUTPUT("Output");

    private final String displayName;

    AudioBlockCategory(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}
