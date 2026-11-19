package com.aresstack.audio.domain;

/** Carry samples and their current format through format-changing pipeline blocks. */
public final class AudioBuffer {

    private final short[] samples;
    private final PcmAudioFormat format;

    public AudioBuffer(short[] samples, PcmAudioFormat format) {
        if (samples == null) {
            throw new IllegalArgumentException("Samples must not be null.");
        }
        if (format == null) {
            throw new IllegalArgumentException("Format must not be null.");
        }
        this.samples = samples;
        this.format = format;
    }

    public short[] getSamples() {
        return samples;
    }

    public PcmAudioFormat getFormat() {
        return format;
    }
}
