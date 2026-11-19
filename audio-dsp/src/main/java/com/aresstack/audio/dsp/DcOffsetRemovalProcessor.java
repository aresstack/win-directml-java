package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Remove the DC component (constant offset) from the signal with a slowly adapting running
 * estimate. Keep the estimate across frames so the correction stays continuous at frame borders.
 */
public final class DcOffsetRemovalProcessor implements Pcm16Processor {

    /** Adaption speed of the offset estimate; small values adapt slowly and distort speech less. */
    private static final double ADAPTION = 0.002d;

    private double offsetEstimate;

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        for (int i = 0; i < sampleCount; i++) {
            double input = samples[i];
            offsetEstimate += ADAPTION * (input - offsetEstimate);
            samples[i] = clamp(input - offsetEstimate);
        }
    }

    private static short clamp(double value) {
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(value);
    }
}
