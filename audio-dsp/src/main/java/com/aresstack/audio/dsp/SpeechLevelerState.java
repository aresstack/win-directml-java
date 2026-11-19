package com.aresstack.audio.dsp;

/**
 * Transient runtime state for the {@link SpeechLevelerProcessor}: the smoothed speech/background levels, the
 * currently applied gain and the hold timer. A fresh instance per run gives reproducible results and never
 * leaks a gain from a previous recording. This is never persisted into a profile.
 */
public final class SpeechLevelerState {

    double speechEnvelope;      // smoothed mean-square of the signal
    double speechLevelDb = Double.NaN;
    double backgroundLevelDb = Double.NaN;
    double currentGainDb;
    int holdSamplesRemaining;
    boolean initialized;
}
