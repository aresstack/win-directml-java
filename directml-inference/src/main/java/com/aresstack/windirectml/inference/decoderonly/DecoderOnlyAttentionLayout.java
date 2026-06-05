package com.aresstack.windirectml.inference.decoderonly;

/**
 * Describes the shared grouped-query-attention head mapping for decoder-only models.
 */
public final class DecoderOnlyAttentionLayout {

    private final int numQueryHeads;
    private final int numKeyValueHeads;
    private final int qHeadsPerKvHead;

    public DecoderOnlyAttentionLayout(int numQueryHeads, int numKeyValueHeads) {
        if (numQueryHeads <= 0) {
            throw new IllegalArgumentException("numQueryHeads must be positive");
        }
        if (numKeyValueHeads <= 0) {
            throw new IllegalArgumentException("numKeyValueHeads must be positive");
        }
        if (numQueryHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException(
                    "numQueryHeads must be divisible by numKeyValueHeads for grouped-query attention");
        }
        this.numQueryHeads = numQueryHeads;
        this.numKeyValueHeads = numKeyValueHeads;
        this.qHeadsPerKvHead = numQueryHeads / numKeyValueHeads;
    }

    public int numQueryHeads() {
        return numQueryHeads;
    }

    public int numKeyValueHeads() {
        return numKeyValueHeads;
    }

    public int qHeadsPerKvHead() {
        return qHeadsPerKvHead;
    }

    /**
     * Map one query head to the KV head that owns its cached K/V vectors.
     */
    public int kvHeadForQueryHead(int queryHead) {
        if (queryHead < 0 || queryHead >= numQueryHeads) {
            throw new IllegalArgumentException("queryHead out of range: " + queryHead);
        }
        int kvHead = queryHead / qHeadsPerKvHead;
        if (kvHead >= numKeyValueHeads) {
            throw new IllegalArgumentException(
                    "queryHead out of range for configured GQA mapping: " + queryHead);
        }
        return kvHead;
    }
}
