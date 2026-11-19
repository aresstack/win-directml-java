package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Measure the signal without changing it: running RMS of the last frame, overall peak magnitude
 * and the number of clipped samples (magnitude at the 16-bit limit). Place it anywhere in the
 * pipeline to observe levels for logging and debugging.
 */
public final class AudioLevelMeter implements Pcm16Processor {

    private volatile double lastFrameRms;
    private volatile int peak;
    /** Peak since the last {@link #consumeRecentPeak()} — the LIVE level for meters/waveforms. */
    private volatile int recentPeak;
    private volatile long clippedSampleCount;
    private volatile long totalSampleCount;
    // Running sum of squares over every sample seen, for the overall RMS across the whole recording.
    private volatile double overallSumOfSquares;

    @Override
    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        double sumOfSquares = 0;
        int framePeak = 0;
        long clipped = clippedSampleCount;
        for (int i = 0; i < sampleCount; i++) {
            int magnitude = Math.abs((int) samples[i]);
            sumOfSquares += (double) magnitude * magnitude;
            if (magnitude > framePeak) {
                framePeak = magnitude;
            }
            if (magnitude >= Short.MAX_VALUE) {
                clipped++;
            }
        }
        lastFrameRms = sampleCount == 0 ? 0 : Math.sqrt(sumOfSquares / sampleCount);
        if (framePeak > peak) {
            peak = framePeak; // the OVERALL maximum keeps its historical meaning (quality checks)
        }
        if (framePeak > recentPeak) {
            recentPeak = framePeak;
        }
        clippedSampleCount = clipped;
        totalSampleCount += sampleCount;
        overallSumOfSquares += sumOfSquares;
    }

    /**
     * The peak since the LAST call, then reset — a rolling window for live level displays. The
     * cumulative {@link #getPeak()} froze every meter at the loudest moment of the recording
     * (one knock on the microphone pinned the bar at maximum forever).
     */
    public int consumeRecentPeak() {
        int value = recentPeak;
        recentPeak = 0;
        return value;
    }

    public double getLastFrameRms() {
        return lastFrameRms;
    }

    /** @return the RMS across every sample processed so far (0 when nothing was measured). */
    public double getOverallRms() {
        long total = totalSampleCount;
        return total == 0 ? 0 : Math.sqrt(overallSumOfSquares / total);
    }

    public int getPeak() {
        return peak;
    }

    public long getClippedSampleCount() {
        return clippedSampleCount;
    }

    public long getTotalSampleCount() {
        return totalSampleCount;
    }

    /** Reset all measurements, e.g. before a new recording. */
    public void reset() {
        lastFrameRms = 0;
        peak = 0;
        recentPeak = 0;
        clippedSampleCount = 0;
        totalSampleCount = 0;
        overallSumOfSquares = 0;
    }

    public String describe() {
        return String.format("rms=%.0f peak=%d clipped=%d samples=%d",
                lastFrameRms, peak, clippedSampleCount, totalSampleCount);
    }
}
