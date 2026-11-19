package com.aresstack.audio.profile;

/** Define every processing function that can be placed in an audio profile. */
public enum AudioBlockType {
    PASSTHROUGH("---"),
    CHANNEL_MIXER("Channel mixer"),
    LOW_PASS("Low-pass filter"),
    HIGH_PASS("High-pass filter"),
    BAND_PASS("Band-pass filter"),
    BAND_STOP("Band-stop filter"),
    RESAMPLER("Resampler"),
    DC_OFFSET_REMOVAL("DC offset removal"),
    NOISE_GATE("Noise gate"),
    COMPRESSOR("Compressor"),
    LIMITER("Limiter"),
    GAIN("Gain"),
    PARAMETRIC_EQ("Parametric Equalizer"),
    LOW_SHELF("Low-Shelf Equalizer"),
    HIGH_SHELF("High-Shelf Equalizer"),
    EQUALIZER("Equalizer"),
    VOICE_ACTIVITY_DETECTION("Voice Activity Detection"),
    SPEECH_GATE("Speech Gate"),
    EXPANDER("Expander"),
    SILENCE_TRIMMER("Silence Trimmer"),
    DE_ESSER("De-Esser"),
    ADAPTIVE_HUM_REMOVAL("Adaptive Hum Removal"),
    PLOSIVE_REDUCTION("Plosive Reduction"),
    BREATH_REDUCTION("Breath Reduction"),
    DE_ESSER_FFT("De-Esser (FFT)"),
    ADAPTIVE_HUM_REMOVAL_FFT("Adaptive Hum Removal (FFT)"),
    PLOSIVE_REDUCTION_FFT("Plosive Reduction (FFT)"),
    BREATH_REDUCTION_FFT("Breath Reduction (FFT)"),
    NOISE_PROFILER("Noise Profiler"),
    ADAPTIVE_NOISE_SUPPRESSION("Adaptive Noise Suppression"),
    SPEECH_LEVELER("Speech Leveler"),
    FINAL_LOUDNESS_NORMALIZER("Final Loudness Normalizer"),
    ROOM_REVERB_ANALYZER("Room/Reverb Analyzer"),
    DEREVERBERATION("Dereverberation"),
    CHANNEL_SELECTOR("Channel Selector"),
    MATRIX_MIXER("Matrix Mixer"),
    CHANNEL_GAIN_POLARITY("Channel Gain and Polarity"),
    PHASE_CORRELATION_ANALYZER("Phase and Correlation Analyzer"),
    CHANNEL_DELAY_ALIGNMENT("Channel Delay Alignment"),
    BEST_CHANNEL_SELECTOR("Best Channel Selector"),
    CHANNEL_HEALTH_ANALYZER("Channel Health Analyzer"),
    MID_SIDE_PROCESSOR("Mid/Side Processor"),
    CENTER_SPEECH_EXTRACTOR("Center Speech Extractor"),
    STEREO_WIDTH_CONTROL("Stereo Width Control"),
    DELAY_AND_SUM_BEAMFORMER("Delay-and-Sum Beamformer"),
    DIRECTION_OF_ARRIVAL_ANALYZER("Direction of Arrival Analyzer"),
    SPEECH_ENHANCER("Speech Enhancer"),
    VOICE_ISOLATION("Voice Isolation");

    private final String displayName;

    AudioBlockType(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }

    @Override
    public String toString() {
        return displayName;
    }
}
