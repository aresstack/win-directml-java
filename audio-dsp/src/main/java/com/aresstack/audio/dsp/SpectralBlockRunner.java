package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Run a short-time spectral modifier over interleaved PCM16, one channel at a time (channel count and rate
 * preserved). A fresh {@link SpectralModifier} is created per channel so cross-frame state stays isolated.
 */
public final class SpectralBlockRunner {

    /** Creates a fresh modifier for one channel. */
    public interface ModifierFactory {
        SpectralModifier create();
    }

    private SpectralBlockRunner() {
    }

    public static void apply(short[] samples, int count, PcmAudioFormat format, int frameSize, int hop,
                             ModifierFactory factory) {
        int channels = Math.max(1, format.getChannels());
        int frames = count / channels;
        if (frames <= 0) {
            return;
        }
        ShortTimeFourierTransform stft =
                new ShortTimeFourierTransform(frameSize, hop, new CommonsMathFourierTransform());
        double[] mono = new double[frames];
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < frames; i++) {
                mono[i] = samples[i * channels + c];
            }
            double[] out = stft.process(mono, format.getSampleRateHz(), factory.create());
            for (int i = 0; i < frames; i++) {
                samples[i * channels + c] = clamp(out[i]);
            }
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
