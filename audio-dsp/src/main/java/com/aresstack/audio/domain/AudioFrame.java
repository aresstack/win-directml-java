package com.aresstack.audio.domain;

/**
 * Carry one fixed-length block of PCM samples plus its format through the processing chain.
 */
public final class AudioFrame {

    private final Pcm16Samples samples;
    private final PcmAudioFormat format;

    public AudioFrame(Pcm16Samples samples, PcmAudioFormat format) {
        if (samples == null) {
            throw new IllegalArgumentException("Samples must not be null.");
        }
        if (format == null) {
            throw new IllegalArgumentException("Format must not be null.");
        }
        this.samples = samples;
        this.format = format;
    }

    public Pcm16Samples getSamples() {
        return samples;
    }

    public PcmAudioFormat getFormat() {
        return format;
    }
}
