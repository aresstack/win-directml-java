package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Run a {@link StatefulAudioBlockProcessor} over a whole mono PCM buffer using windowed overlap-add: split
 * the signal into overlapping frames, window each, process it, and add the results back with per-sample
 * window-sum normalization so no samples are lost and an identity processor reconstructs the input exactly
 * (within 16-bit rounding), including at the edges.
 *
 * <p>This is the framing infrastructure adaptive/spectral blocks (later phases) build on; it keeps the
 * per-frame contract out of each block. The current static blocks do not need it and process at buffer
 * level directly.</p>
 */
public final class OverlapAddFramer {

    private final int frameSize;
    private final int hop;
    private final double[] window;

    public OverlapAddFramer(int frameSize, int hop) {
        if (frameSize <= 0) {
            throw new IllegalArgumentException("Frame size must be positive.");
        }
        if (hop <= 0 || hop > frameSize) {
            throw new IllegalArgumentException("Hop must be in (0, frameSize].");
        }
        this.frameSize = frameSize;
        this.hop = hop;
        this.window = hannWindow(frameSize);
    }

    public int getFrameSize() {
        return frameSize;
    }

    public int getHop() {
        return hop;
    }

    /**
     * @param mono      the input samples (mono)
     * @param format    the format handed to the processor's {@link StatefulAudioBlockProcessor#initialize}
     * @param processor the frame processor to run; it is initialized here and reset is the caller's concern
     * @return a new buffer of the same length as {@code mono}
     */
    public short[] process(short[] mono, PcmAudioFormat format, StatefulAudioBlockProcessor processor) {
        if (mono == null) {
            throw new IllegalArgumentException("Samples must not be null.");
        }
        int length = mono.length;
        short[] result = new short[length];
        if (length == 0) {
            return result;
        }
        processor.initialize(format);

        double[] accumulator = new double[length];
        double[] normalization = new double[length];
        short[] frameIn = new short[frameSize];
        short[] frameOut = new short[frameSize];

        for (int start = 0; start < length; start += hop) {
            int copy = Math.min(frameSize, length - start);
            for (int j = 0; j < frameSize; j++) {
                frameIn[j] = j < copy ? mono[start + j] : 0;
                frameOut[j] = 0;
            }
            processor.process(frameIn, frameOut);
            for (int j = 0; j < copy; j++) {
                double w = window[j];
                accumulator[start + j] += w * frameOut[j];
                normalization[start + j] += w;
            }
        }

        for (int i = 0; i < length; i++) {
            double value = normalization[i] > 1.0e-9d ? accumulator[i] / normalization[i] : accumulator[i];
            result[i] = clamp(value);
        }
        return result;
    }

    private static double[] hannWindow(int size) {
        double[] w = new double[size];
        if (size == 1) {
            w[0] = 1.0d;
            return w;
        }
        for (int i = 0; i < size; i++) {
            // Add a small floor so no window sample is exactly zero; keeps the normalization well-defined.
            w[i] = 0.5d - 0.5d * Math.cos(2.0d * Math.PI * i / (size - 1)) + 1.0e-3d;
        }
        return w;
    }

    private static short clamp(double value) {
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(value);
    }
}
