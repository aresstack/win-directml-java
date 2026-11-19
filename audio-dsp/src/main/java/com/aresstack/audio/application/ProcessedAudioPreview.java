package com.aresstack.audio.application;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * The result of processing a test source through a pipeline snapshot: the processed samples and their
 * format, plus provenance (which source and which pipeline signature produced it) so the UI can tell
 * whether the result still matches the current editor pipeline.
 */
public final class ProcessedAudioPreview {

    private final short[] samples;
    private final PcmAudioFormat format;
    private final long durationMillis;
    private final String sourceId;
    private final String pipelineSignature;

    public ProcessedAudioPreview(short[] samples, PcmAudioFormat format, long durationMillis,
                                 String sourceId, String pipelineSignature) {
        if (samples == null) {
            throw new IllegalArgumentException("Samples must not be null.");
        }
        if (format == null) {
            throw new IllegalArgumentException("Format must not be null.");
        }
        this.samples = samples;
        this.format = format;
        this.durationMillis = durationMillis;
        this.sourceId = sourceId == null ? "" : sourceId;
        this.pipelineSignature = pipelineSignature == null ? "" : pipelineSignature;
    }

    public short[] getSamples() {
        return samples;
    }

    public PcmAudioFormat getFormat() {
        return format;
    }

    public long getDurationMillis() {
        return durationMillis;
    }

    public String getSourceId() {
        return sourceId;
    }

    public String getPipelineSignature() {
        return pipelineSignature;
    }
}
