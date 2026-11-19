package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Process one block of 16-bit PCM samples in place. Implementations may keep state between
 * frames (filters, envelopes) but must not perform any I/O and must not write output anywhere —
 * the pipeline and sinks handle data flow.
 */
public interface Pcm16Processor {

    /**
     * Process {@code sampleCount} samples of the given array in place.
     *
     * @param samples     the sample buffer; entries beyond {@code sampleCount} must be ignored
     * @param sampleCount the number of valid samples
     * @param format      the PCM format the samples are in
     */
    void process(short[] samples, int sampleCount, PcmAudioFormat format);
}
