package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Attenuate low-frequency rumble (footsteps, desk knocks, wind) below the cutoff with a
 * first-order IIR high-pass filter. Keep the filter state across frames.
 *
 * <p>Difference equation: {@code y[n] = a * (y[n-1] + x[n] - x[n-1])} with
 * {@code a = RC / (RC + dt)} and {@code RC = 1 / (2 * pi * cutoff)}.</p>
 */
public final class HighPassFilterProcessor implements Pcm16Processor {

    private final double cutoffHz;

    private double coefficient;
    private int coefficientSampleRate;
    private double previousInput;
    private double previousOutput;

    public HighPassFilterProcessor(double cutoffHz) {
        if (cutoffHz <= 0) {
            throw new IllegalArgumentException("Cutoff frequency must be positive: " + cutoffHz);
        }
        this.cutoffHz = cutoffHz;
    }

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        updateCoefficient(format.getSampleRateHz());
        for (int i = 0; i < sampleCount; i++) {
            double input = samples[i];
            double output = coefficient * (previousOutput + input - previousInput);
            previousInput = input;
            previousOutput = output;
            samples[i] = clamp(output);
        }
    }

    /** Recompute the filter coefficient when the sample rate is first seen or changes. */
    private void updateCoefficient(int sampleRateHz) {
        if (sampleRateHz == coefficientSampleRate) {
            return;
        }
        double rc = 1.0d / (2.0d * Math.PI * cutoffHz);
        double dt = 1.0d / sampleRateHz;
        coefficient = rc / (rc + dt);
        coefficientSampleRate = sampleRateHz;
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
