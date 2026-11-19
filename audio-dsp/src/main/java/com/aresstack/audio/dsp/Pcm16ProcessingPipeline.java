package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Call a fixed, ordered list of {@link Pcm16Processor}s on every block. Extend the pipeline by
 * passing more processors — never by modifying this class.
 */
public final class Pcm16ProcessingPipeline implements AudioProcessingPipeline {

    private final List<Pcm16Processor> processors;

    public Pcm16ProcessingPipeline(List<Pcm16Processor> processors) {
        if (processors == null) {
            throw new IllegalArgumentException("Processor list must not be null.");
        }
        this.processors = Collections.unmodifiableList(new ArrayList<Pcm16Processor>(processors));
    }

    public List<Pcm16Processor> getProcessors() {
        return processors;
    }

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        for (int i = 0; i < processors.size(); i++) {
            processors.get(i).process(samples, sampleCount, format);
        }
    }
}
