package com.aresstack.audio.domain;

/**
 * Describe a PCM audio format: sample rate, channel count and bits per sample.
 * Samples are always treated as 16-bit signed little endian throughout this module.
 */
public final class PcmAudioFormat {

    private final int sampleRateHz;
    private final int channels;
    private final int bitsPerSample;

    public PcmAudioFormat(int sampleRateHz, int channels, int bitsPerSample) {
        if (sampleRateHz <= 0) {
            throw new IllegalArgumentException("Sample rate must be positive: " + sampleRateHz);
        }
        if (channels <= 0) {
            throw new IllegalArgumentException("Channel count must be positive: " + channels);
        }
        if (bitsPerSample != 16) {
            throw new IllegalArgumentException(
                    "Only 16-bit signed PCM is supported, got " + bitsPerSample + " bits per sample.");
        }
        this.sampleRateHz = sampleRateHz;
        this.channels = channels;
        this.bitsPerSample = bitsPerSample;
    }

    /** Create the default speech format: 16 kHz, mono, 16-bit signed little endian. */
    public static PcmAudioFormat speechDefault() {
        return new PcmAudioFormat(16000, 1, 16);
    }

    public int getSampleRateHz() {
        return sampleRateHz;
    }

    public int getChannels() {
        return channels;
    }

    public int getBitsPerSample() {
        return bitsPerSample;
    }

    public int getBytesPerSample() {
        return bitsPerSample / 8;
    }

    /** Compute how many samples (per channel set) make up a frame of the given duration. */
    public int samplesForMillis(int millis) {
        return (int) ((long) sampleRateHz * channels * millis / 1000L);
    }

    public String toString() {
        return sampleRateHz + " Hz, " + channels + " channel(s), " + bitsPerSample + "-bit PCM";
    }
}
