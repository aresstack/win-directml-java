package com.aresstack.windirectml.encoder.pooling;

/**
 * CPU-Referenzimplementierungen für Mean Pooling über Token-Embeddings.
 * <p>
 * Identische Semantik wie die spätere GPU-Variante: padding-Tokens
 * (Attention-Mask = 0) werden ignoriert. Stehen den Tests als
 * Vergleichswert zur Verfügung.
 */
public final class MeanPooling {

    private MeanPooling() {
    }

    /**
     * Mean Pooling über die Sequenz-Dimension.
     *
     * @param tokenEmbeddings dichtes Array, Form {@code [seq, hidden]}.
     * @param attentionMask   Werte 0 oder 1, Länge {@code seq}.
     * @return Vektor der Länge {@code hidden}.
     */
    public static float[] pool(float[][] tokenEmbeddings, int[] attentionMask) {
        if (tokenEmbeddings.length == 0) {
            throw new IllegalArgumentException("tokenEmbeddings must not be empty");
        }
        int seq = tokenEmbeddings.length;
        int hidden = tokenEmbeddings[0].length;
        if (attentionMask.length != seq) {
            throw new IllegalArgumentException(
                    "attentionMask length " + attentionMask.length + " != seq " + seq);
        }
        float[] sum = new float[hidden];
        long count = 0;
        for (int t = 0; t < seq; t++) {
            if (attentionMask[t] == 0) continue;
            float[] row = tokenEmbeddings[t];
            if (row.length != hidden) {
                throw new IllegalArgumentException("inconsistent row length at " + t);
            }
            for (int h = 0; h < hidden; h++) sum[h] += row[h];
            count++;
        }
        if (count == 0) return sum; // alle Tokens maskiert
        float inv = (float) (1.0 / count);
        for (int h = 0; h < hidden; h++) sum[h] *= inv;
        return sum;
    }
}

