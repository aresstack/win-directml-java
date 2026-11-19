package com.aresstack.audio.application;

import com.aresstack.audio.domain.PcmAudioFormat;

import java.io.File;

/**
 * The outcome of normalizing a raw recording into the canonical speech WAV: the written file, the
 * source and target formats, the duration and the level statistics measured on the <em>raw</em>
 * captured signal (so clipping is detected before the limiter would hide it). Immutable.
 */
public final class NormalizationResult {

    private final File targetFile;
    private final PcmAudioFormat sourceFormat;
    private final PcmAudioFormat targetFormat;
    private final long durationMillis;
    private final double overallRms;
    private final int peak;
    private final long clippedSamples;
    private final long totalSamples;

    public NormalizationResult(File targetFile, PcmAudioFormat sourceFormat, PcmAudioFormat targetFormat,
                               long durationMillis, double overallRms, int peak, long clippedSamples,
                               long totalSamples) {
        this.targetFile = targetFile;
        this.sourceFormat = sourceFormat;
        this.targetFormat = targetFormat;
        this.durationMillis = durationMillis;
        this.overallRms = overallRms;
        this.peak = peak;
        this.clippedSamples = clippedSamples;
        this.totalSamples = totalSamples;
    }

    public File getTargetFile() {
        return targetFile;
    }

    public PcmAudioFormat getSourceFormat() {
        return sourceFormat;
    }

    public PcmAudioFormat getTargetFormat() {
        return targetFormat;
    }

    public long getDurationMillis() {
        return durationMillis;
    }

    public double getOverallRms() {
        return overallRms;
    }

    public int getPeak() {
        return peak;
    }

    public long getClippedSamples() {
        return clippedSamples;
    }

    public long getTotalSamples() {
        return totalSamples;
    }
}
