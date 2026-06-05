package com.aresstack.windirectml.inference.decoderonly;

/**
 * Defines the canonical decoder-only KV-cache memory layout.
 *
 * <p>Layout: {@code [kvHead][position][headDim]} row-major. This matches the
 * existing Qwen CPU cache and the GPU-resident Direct3D buffer layout.</p>
 */
public final class DecoderOnlyKvCacheLayout {

    private final int numLayers;
    private final int numKeyValueHeads;
    private final int maxSequenceLength;
    private final int headDim;

    public DecoderOnlyKvCacheLayout(int numLayers, int numKeyValueHeads,
                                    int maxSequenceLength, int headDim) {
        requirePositive("numLayers", numLayers);
        requirePositive("numKeyValueHeads", numKeyValueHeads);
        requirePositive("maxSequenceLength", maxSequenceLength);
        requirePositive("headDim", headDim);
        this.numLayers = numLayers;
        this.numKeyValueHeads = numKeyValueHeads;
        this.maxSequenceLength = maxSequenceLength;
        this.headDim = headDim;
    }

    public int numLayers() {
        return numLayers;
    }

    public int numKeyValueHeads() {
        return numKeyValueHeads;
    }

    public int maxSequenceLength() {
        return maxSequenceLength;
    }

    public int headDim() {
        return headDim;
    }

    public long bytesPerLayer() {
        return (long) numKeyValueHeads * maxSequenceLength * headDim * Float.BYTES;
    }

    public long totalKeyValueBytes() {
        return 2L * numLayers * bytesPerLayer();
    }

    /**
     * Return the flat index for {@code [kvHead][position][headOffset]}.
     */
    public int flatIndex(int kvHead, int position, int headOffset) {
        validateRange("kvHead", kvHead, numKeyValueHeads);
        validateRange("position", position, maxSequenceLength);
        validateRange("headOffset", headOffset, headDim);
        return ((kvHead * maxSequenceLength) + position) * headDim + headOffset;
    }

    public int validPrefixLength(int sequenceLength) {
        if (sequenceLength < 0 || sequenceLength > maxSequenceLength) {
            throw new IllegalArgumentException(
                    "sequenceLength=" + sequenceLength + " out of [0.." + maxSequenceLength + "]");
        }
        return sequenceLength * headDim;
    }

    private static void requirePositive(String name, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive");
        }
    }

    private static void validateRange(String name, int value, int exclusiveUpperBound) {
        if (value < 0 || value >= exclusiveUpperBound) {
            throw new IllegalArgumentException(
                    name + "=" + value + " out of [0.." + (exclusiveUpperBound - 1) + "]");
        }
    }
}
