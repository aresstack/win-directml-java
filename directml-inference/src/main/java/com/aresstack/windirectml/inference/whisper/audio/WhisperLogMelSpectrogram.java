package com.aresstack.windirectml.inference.whisper.audio;

import java.util.Arrays;

/**
 * Convert 16-kHz mono float PCM into the 30-second Whisper audio-encoder input.
 * Follow OpenAI Whisper's centered STFT, periodic Hann window, Slaney Mel bank,
 * log10 floor, 8-decade dynamic range, and final (log + 4) / 4 scaling.
 *
 * Input is the raw audio waveform, not a pre-emphasized or denoised signal.
 * The returned matrix has shape [nMels][3000].
 */
public final class WhisperLogMelSpectrogram {
    public static final int SAMPLE_RATE = 16000;
    public static final int FFT_SIZE = 400;
    public static final int HOP_LENGTH = 160;
    public static final int SAMPLES_PER_CHUNK = SAMPLE_RATE * 30;
    public static final int FRAMES_PER_CHUNK = SAMPLES_PER_CHUNK / HOP_LENGTH;

    private final int melBands;
    private final float[][] filters;
    private final FourierTransform fft;
    private final double[] window;

    public WhisperLogMelSpectrogram(int melBands) {
        this.melBands = melBands;
        this.filters = WhisperMelFilterBank.create(melBands);
        this.fft = new BluesteinFourierTransform(FFT_SIZE);
        this.window = new double[FFT_SIZE];
        for (int i = 0; i < FFT_SIZE; i++) {
            window[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / FFT_SIZE);
        }
    }

    /**
     * Pad or trim the selected audio to thirty seconds, and compute its log-Mel features.
     * Keep offsets in sample units so the transcription layer can attach timestamps.
     */
    public float[][] compute(float[] samples, int startSample) {
        if (samples == null || startSample < 0 || startSample > samples.length) {
            throw new IllegalArgumentException("Invalid Whisper audio sample interval");
        }
        float[] audio = new float[SAMPLES_PER_CHUNK];
        int count = Math.min(audio.length, samples.length - startSample);
        System.arraycopy(samples, startSample, audio, 0, count);
        return computeChunk(audio);
    }

    public float[][] compute(float[] samples) {
        return compute(samples, 0);
    }

    private float[][] computeChunk(float[] samples) {
        float[][] result = new float[melBands][FRAMES_PER_CHUNK];
        double[] real = new double[FFT_SIZE];
        double[] imag = new double[FFT_SIZE];
        double[] powers = new double[FFT_SIZE / 2 + 1];
        double maxLog = Double.NEGATIVE_INFINITY;
        for (int frame = 0; frame < FRAMES_PER_CHUNK; frame++) {
            int center = frame * HOP_LENGTH;
            for (int i = 0; i < FFT_SIZE; i++) {
                int sample = reflect(center + i - FFT_SIZE / 2, samples.length);
                real[i] = samples[sample] * window[i];
                imag[i] = 0.0;
            }
            fft.forward(real, imag);
            for (int bin = 0; bin < powers.length; bin++) {
                powers[bin] = real[bin] * real[bin] + imag[bin] * imag[bin];
            }
            for (int mel = 0; mel < melBands; mel++) {
                double energy = 0.0;
                float[] weights = filters[mel];
                for (int bin = 0; bin < powers.length; bin++) {
                    energy += weights[bin] * powers[bin];
                }
                float log = (float) Math.log10(Math.max(energy, 1.0e-10));
                result[mel][frame] = log;
                maxLog = Math.max(maxLog, log);
            }
        }
        double floor = maxLog - 8.0;
        for (int mel = 0; mel < melBands; mel++) {
            for (int frame = 0; frame < FRAMES_PER_CHUNK; frame++) {
                result[mel][frame] = (float) ((Math.max(result[mel][frame], floor) + 4.0) / 4.0);
            }
        }
        return result;
    }

    private static int reflect(int position, int length) {
        if (position < 0) {
            return -position;
        }
        if (position >= length) {
            return 2 * length - 2 - position;
        }
        return position;
    }
}
