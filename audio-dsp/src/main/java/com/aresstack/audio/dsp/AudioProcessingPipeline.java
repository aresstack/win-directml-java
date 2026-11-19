package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Run a sequence of DSP steps over sample blocks. Implementations know nothing about Java Sound,
 * files or the network — they only transform samples.
 */
public interface AudioProcessingPipeline {

    /** Apply every processing step, in order, to the given samples in place. */
    void process(short[] samples, int sampleCount, PcmAudioFormat format);
}
