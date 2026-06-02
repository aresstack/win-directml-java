package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.encoder.PoolingStrategy;

/**
 * CPU helper that prepares the pre-normalised pooling weight vector
 * the GPU side of mean-pooling consumes.
 * <p>
 * For {@link PoolingStrategy#MEAN}: {@code w[t] = mask[t] / Σmask}
 * over the valid token prefix; padded positions stay zero. The GPU
 * then performs a single GEMM ({@code [1,seq] · [seq,H] → [1,H]})
 * instead of a separate reduce + divide – this keeps the pooling on
 * FL 1.0 baseline and avoids a per-call elementwise dispatch.
 * <p>
 * For non-mean strategies this helper currently throws; CLS/MAX
 * support lands together with their respective DirectML kernels.
 */
public final class BertPoolingWeights {

    private BertPoolingWeights() {
    }

    /**
     * Build the pooling weight vector for {@link PoolingStrategy#MEAN}.
     *
     * @param attentionMask binary mask of shape {@code [seqLen]} (0/1).
     * @param seqLen        number of real tokens in {@code attentionMask}.
     * @param seqBucket     padded length the GPU stack expects
     *                      ({@code >= seqLen}); trailing entries stay zero.
     * @return weights array of length {@code seqBucket}.
     * @throws IllegalStateException if the mask is all zero.
     */
    public static float[] mean(int[] attentionMask, int seqLen, int seqBucket) {
        if (attentionMask == null) {
            throw new IllegalArgumentException("attentionMask must not be null");
        }
        if (seqLen <= 0 || seqBucket < seqLen) {
            throw new IllegalArgumentException(
                    "invalid lengths: seqLen=" + seqLen + ", seqBucket=" + seqBucket);
        }
        int validCount = 0;
        for (int i = 0; i < seqLen; i++) {
            if (attentionMask[i] != 0) validCount++;
        }
        if (validCount == 0) {
            throw new IllegalStateException("Attention mask is all zero – nothing to pool");
        }
        float inv = 1.0f / validCount;
        float[] w = new float[seqBucket];
        for (int i = 0; i < seqLen; i++) {
            if (attentionMask[i] != 0) w[i] = inv;
        }
        return w;
    }

    /**
     * Build the additive attention mask consumed by the encoder stack:
     * {@code 0.0f} for valid tokens, {@code -1e9f} elsewhere (both for
     * tokens where {@code attentionMask[i] == 0} and for padded bucket
     * positions {@code [seqLen, seqBucket)}).
     */
    public static float[] additiveMask(int[] attentionMask, int seqLen, int seqBucket) {
        if (attentionMask == null) {
            throw new IllegalArgumentException("attentionMask must not be null");
        }
        if (seqLen <= 0 || seqBucket < seqLen) {
            throw new IllegalArgumentException(
                    "invalid lengths: seqLen=" + seqLen + ", seqBucket=" + seqBucket);
        }
        float[] mask = new float[seqBucket];
        for (int i = 0; i < seqLen; i++) {
            mask[i] = attentionMask[i] == 0 ? -1e9f : 0f;
        }
        for (int i = seqLen; i < seqBucket; i++) {
            mask[i] = -1e9f;
        }
        return mask;
    }
}

