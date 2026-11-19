package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Frame-based processor for adaptive/spectral blocks that keep internal state between frames (noise model,
 * adaptive coefficients, FFT history, level smoothing). Runs off the Swing EDT.
 *
 * <p>The contract: {@link #initialize(PcmAudioFormat)} once before the first frame, {@link #process}
 * per fixed-size frame (mutating {@code output} in place; {@code input} and {@code output} have the same
 * length), and {@link #reset()} to drop all learned state so the same processor can be reused for a new
 * recording. Frames are 16-bit signed PCM; framing/windowing/overlap-add is provided by
 * {@link OverlapAddFramer}, so implementations only see one frame at a time.</p>
 */
public interface StatefulAudioBlockProcessor {

    /** Prepare internal buffers for the given format; called once before processing. */
    void initialize(PcmAudioFormat format);

    /** Process one frame; write the result into {@code output} (same length as {@code input}). */
    void process(short[] input, short[] output);

    /** Drop all learned/adaptive state so the processor can be reused for a fresh recording. */
    void reset();
}
