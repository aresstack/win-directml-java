package com.aresstack.audio.domain;

/**
 * Hold a block of 16-bit PCM samples together with the number of valid entries.
 * The backing array may be larger than the valid range; never read past {@link #getCount()}.
 */
public final class Pcm16Samples {

    private final short[] samples;
    private final int count;

    public Pcm16Samples(short[] samples, int count) {
        if (samples == null) {
            throw new IllegalArgumentException("Sample array must not be null.");
        }
        if (count < 0 || count > samples.length) {
            throw new IllegalArgumentException(
                    "Sample count " + count + " is outside the array of length " + samples.length + ".");
        }
        this.samples = samples;
        this.count = count;
    }

    public short[] getSamples() {
        return samples;
    }

    public int getCount() {
        return count;
    }

    /** Copy the valid range into a right-sized array. */
    public short[] toArray() {
        short[] copy = new short[count];
        System.arraycopy(samples, 0, copy, 0, count);
        return copy;
    }
}
