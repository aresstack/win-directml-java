package com.aresstack.windirectml.inference.piper;

import java.util.Objects;

/** Hold one validated mono waveform produced by a Piper voice. */
public final class PiperAudio {

    private final int sampleRate;
    private final float[] samples;

    public PiperAudio(int sampleRate, float[] samples) {
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("Piper sample rate must be positive");
        }
        Objects.requireNonNull(samples, "samples");
        if (samples.length == 0) {
            throw new IllegalArgumentException("Piper waveform must contain audio samples");
        }
        for (float sample : samples) {
            if (!Float.isFinite(sample) || sample < -1.001f || sample > 1.001f) {
                throw new IllegalArgumentException("Piper waveform contains invalid PCM samples");
            }
        }
        this.sampleRate = sampleRate;
        this.samples = samples.clone();
    }

    public int sampleRate() {
        return sampleRate;
    }

    public int sampleCount() {
        return samples.length;
    }

    public double durationSeconds() {
        return (double) samples.length / sampleRate;
    }

    public float[] samples() {
        return samples.clone();
    }
}
