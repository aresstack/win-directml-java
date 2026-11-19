package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Apply a fixed level change: {@code output = input * 10^(dB/20)}, clamped to the PCM-16 range so a boost
 * can never wrap around. {@code 0 dB} is a bit-exact pass-through. This is a plain gain only — no compression
 * or limiting.
 */
public final class GainProcessor implements Pcm16Processor {

    private final double factor;

    public GainProcessor(double gainDb) {
        double linear = Double.isNaN(gainDb) || Double.isInfinite(gainDb)
                ? 1.0d : Math.pow(10.0d, gainDb / 20.0d);
        this.factor = linear;
    }

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        if (factor == 1.0d) {
            return; // 0 dB: leave every sample untouched (bit-identical)
        }
        for (int i = 0; i < sampleCount; i++) {
            samples[i] = clamp(samples[i] * factor);
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
