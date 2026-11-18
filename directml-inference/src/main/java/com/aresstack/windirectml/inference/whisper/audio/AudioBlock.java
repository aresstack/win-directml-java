package com.aresstack.windirectml.inference.whisper.audio;

/**
 * A block of stereo sound in float: two arrays of a fixed capacity, of which the first
 * {@link #frames()} are in use. A graph renders one block at a time; how many frames that is
 * may change from block to block, but never exceeds the capacity given to {@code prepare}.
 */
public final class AudioBlock {
    public final float[] left;
    public final float[] right;
    private int frames;

    public AudioBlock(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("A block holds at least one frame.");
        }
        left = new float[capacity];
        right = new float[capacity];
        frames = capacity;
    }

    public int capacity() {
        return left.length;
    }

    /** How many frames of this block are in use. */
    public int frames() {
        return frames;
    }

    public void setFrames(int frames) {
        if (frames < 0 || frames > left.length) {
            throw new IllegalArgumentException("A block of " + left.length + " frames cannot hold " + frames + ".");
        }
        this.frames = frames;
    }

    /** Sets the frames in use to silence. */
    public void clear() {
        clear(0, frames);
    }

    /** Sets {@code count} frames from {@code from} on to silence. */
    public void clear(int from, int count) {
        java.util.Arrays.fill(left, from, from + count, 0f);
        java.util.Arrays.fill(right, from, from + count, 0f);
    }

    /** Copies the frames in use of another block into this one; both then use as many frames. */
    public void copyFrom(AudioBlock other) {
        frames = other.frames;
        System.arraycopy(other.left, 0, left, 0, frames);
        System.arraycopy(other.right, 0, right, 0, frames);
    }

    /**
     * Adds the first {@code count} frames of this block, each side with its own gain, to a bus.
     * This is how a channel reaches a bus: {@code bus += this · gain}, sample by sample.
     */
    public void addTo(AudioBlock bus, int count, float gainLeft, float gainRight) {
        float[] busLeft = bus.left;
        float[] busRight = bus.right;
        for (int i = 0; i < count; i++) {
            busLeft[i] += left[i] * gainLeft;
            busRight[i] += right[i] * gainRight;
        }
    }

    /** Multiplies the frames in use by a gain. */
    public void scale(float gain) {
        for (int i = 0; i < frames; i++) {
            left[i] *= gain;
            right[i] *= gain;
        }
    }
}
