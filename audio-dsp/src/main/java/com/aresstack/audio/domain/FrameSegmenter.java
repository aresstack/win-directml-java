package com.aresstack.audio.domain;

/**
 * Cut an arbitrary incoming sample stream into fixed-size frames. Java Sound delivers whatever
 * buffer sizes it likes; the DSP pipeline wants exact 10/20 ms frames. Keep the remainder between
 * calls and emit it via {@link #flush(FrameConsumer)} at the end of a recording.
 */
public final class FrameSegmenter {

    /** Receive one completed frame; the array is reused, so consume it before returning. */
    public interface FrameConsumer {
        void onFrame(short[] samples, int count);
    }

    private final short[] frame;
    private int filled;

    public FrameSegmenter(int frameSizeSamples) {
        if (frameSizeSamples <= 0) {
            throw new IllegalArgumentException("Frame size must be positive: " + frameSizeSamples);
        }
        this.frame = new short[frameSizeSamples];
    }

    /** Append samples and emit every completed fixed-size frame to the consumer. */
    public void push(short[] samples, int count, FrameConsumer consumer) {
        int offset = 0;
        while (offset < count) {
            int toCopy = Math.min(frame.length - filled, count - offset);
            System.arraycopy(samples, offset, frame, filled, toCopy);
            filled += toCopy;
            offset += toCopy;
            if (filled == frame.length) {
                consumer.onFrame(frame, frame.length);
                filled = 0;
            }
        }
    }

    /** Emit the remaining partial frame, if any, and reset. */
    public void flush(FrameConsumer consumer) {
        if (filled > 0) {
            consumer.onFrame(frame, filled);
            filled = 0;
        }
    }
}
