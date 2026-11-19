package com.aresstack.audio.infrastructure;

/**
 * Convert between 16-bit signed little-endian byte streams (as Java Sound and WAV use them)
 * and {@code short[]} sample buffers.
 */
public final class Pcm16LittleEndianCodec {

    private Pcm16LittleEndianCodec() {
    }

    /** Decode little-endian bytes into samples. {@code byteCount} must be even. */
    public static int decode(byte[] bytes, int byteCount, short[] target) {
        if (byteCount % 2 != 0) {
            throw new IllegalArgumentException("Byte count must be even for 16-bit PCM: " + byteCount);
        }
        int sampleCount = byteCount / 2;
        if (target.length < sampleCount) {
            throw new IllegalArgumentException("Target array too small: need " + sampleCount
                    + " samples, got " + target.length + ".");
        }
        for (int i = 0; i < sampleCount; i++) {
            int low = bytes[2 * i] & 0xFF;
            int high = bytes[2 * i + 1];
            target[i] = (short) ((high << 8) | low);
        }
        return sampleCount;
    }

    /** Encode samples into little-endian bytes. The target must hold {@code 2 * sampleCount} bytes. */
    public static int encode(short[] samples, int sampleCount, byte[] target) {
        int byteCount = sampleCount * 2;
        if (target.length < byteCount) {
            throw new IllegalArgumentException("Target array too small: need " + byteCount
                    + " bytes, got " + target.length + ".");
        }
        for (int i = 0; i < sampleCount; i++) {
            target[2 * i] = (byte) (samples[i] & 0xFF);
            target[2 * i + 1] = (byte) ((samples[i] >> 8) & 0xFF);
        }
        return byteCount;
    }
}
