package com.aresstack.audio.dsp;

/**
 * Mutable per-run state of {@link VoiceActivityDetector}: the adaptive noise-floor estimate, the smoothed
 * speech probability and the state-machine counters. A fresh instance is created for every independent run,
 * and {@link #reset()} restores the initial state so a detector can be reused reproducibly.
 */
public final class VoiceActivityDetectorState {

    boolean initialized;
    double noiseFloorLinear;
    double smoothedProbability;
    int speechRunFrames;
    int silenceRunFrames;
    int hangoverRemainingFrames;
    boolean speechActive;

    public void reset() {
        initialized = false;
        noiseFloorLinear = 0.0d;
        smoothedProbability = 0.0d;
        speechRunFrames = 0;
        silenceRunFrames = 0;
        hangoverRemainingFrames = 0;
        speechActive = false;
    }
}
