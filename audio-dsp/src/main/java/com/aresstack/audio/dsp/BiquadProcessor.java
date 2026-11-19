package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Shared biquad filtering for the peaking and shelving equalizers. One coefficient set (computed from the
 * running sample rate via a {@link BiquadDesign}) is applied per channel with independent transposed
 * Direct-Form-II state, then clamped back to PCM-16. Processing is in float; a fresh processor is created
 * per pipeline run, so state never leaks. Invalid parameters (non-finite/unstable coefficients, frequency
 * at/above Nyquist for the current rate) make the block bypass rather than emit NaN/Infinity or crash.
 */
public class BiquadProcessor implements Pcm16Processor {

    /** Produce coefficients for the actual sample rate seen at processing time. */
    public interface BiquadDesign {
        BiquadCoefficients design(int sampleRateHz);
    }

    private final BiquadDesign design;

    private BiquadCoefficients coefficients;
    private BiquadFilterState[] states;
    private int configuredSampleRate;
    private int configuredChannels;
    private boolean bypassedForCurrentFormat;

    public BiquadProcessor(BiquadDesign design) {
        if (design == null) {
            throw new IllegalArgumentException("Biquad design must not be null.");
        }
        this.design = design;
    }

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        configureFor(format);
        if (bypassedForCurrentFormat) {
            return;
        }
        int channels = format.getChannels();
        for (int i = 0; i < sampleCount; i++) {
            int channel = channels == 1 ? 0 : i % channels;
            samples[i] = clamp(filter(states[channel], samples[i]));
        }
    }

    private double filter(BiquadFilterState state, double x) {
        double y = coefficients.b0 * x + state.z1;
        state.z1 = coefficients.b1 * x - coefficients.a1 * y + state.z2;
        state.z2 = coefficients.b2 * x - coefficients.a2 * y;
        return y;
    }

    private void configureFor(PcmAudioFormat format) {
        if (states != null && configuredSampleRate == format.getSampleRateHz()
                && configuredChannels == format.getChannels()) {
            return;
        }
        configuredSampleRate = format.getSampleRateHz();
        configuredChannels = format.getChannels();
        try {
            coefficients = design.design(configuredSampleRate);
            bypassedForCurrentFormat = false;
        } catch (RuntimeException invalid) {
            // Unsafe parameters for this rate: pass audio through unchanged instead of destabilizing it.
            coefficients = null;
            bypassedForCurrentFormat = true;
            states = new BiquadFilterState[0];
            return;
        }
        states = new BiquadFilterState[configuredChannels];
        for (int i = 0; i < states.length; i++) {
            states[i] = new BiquadFilterState();
        }
    }

    private static short clamp(double value) {
        if (Double.isNaN(value)) {
            return 0;
        }
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(value);
    }
}
