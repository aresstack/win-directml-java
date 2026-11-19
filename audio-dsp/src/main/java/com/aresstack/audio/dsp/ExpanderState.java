package com.aresstack.audio.dsp;

/**
 * Mutable per-run state of {@link ExpanderProcessor}: the smoothed detector envelope, the current applied
 * gain and the hold counter. A fresh instance is created for every independent run; {@link #reset()}
 * restores the initial state so a processor can be reused reproducibly.
 */
public final class ExpanderState {

    boolean initialized;
    double envelopeLinear;
    double gainDb;
    int holdRemainingSamples;

    public void reset() {
        initialized = false;
        envelopeLinear = 0.0d;
        gainDb = 0.0d;
        holdRemainingSamples = 0;
    }
}
