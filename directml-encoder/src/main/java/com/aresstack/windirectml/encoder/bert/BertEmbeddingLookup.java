package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.encoder.EncoderTokenizer;

import java.util.Objects;

/**
 * CPU helper that materialises the BERT-style input embedding sum
 * ({@code word + position + tokenType}) into a packed
 * {@code [seqBucket * hiddenSize]} row-major float array.
 * <p>
 * The lookup is intentionally CPU-side: it is a pure memory gather and
 * doing it on the device would require a {@code DML_OPERATOR_GATHER}
 * (FL 5.0+) which is not available on every shipping in-box
 * {@code DirectML.dll}. By keeping it on the CPU the upload is the
 * single point at which a {@code [B,H]} F32 buffer crosses PCIe.
 * <p>
 * Rows {@code [seqLen, seqBucket)} stay zero – padded positions are
 * masked out later in attention and pooling, so their content does not
 * affect the result. The caller owns the output array; pass it pre-zeroed
 * or use {@link #lookup(BertEncoderConfig, EncoderTokenizer.Encoded, int)}.
 *
 * <p><b>Embedding table layout</b> (HuggingFace BERT-style, row-major):
 * <ul>
 *   <li>{@code wordEmbeddings}      shape {@code [vocabSize, H]}</li>
 *   <li>{@code positionEmbeddings}  shape {@code [maxPositionEmbeddings, H]}</li>
 *   <li>{@code tokenTypeEmbeddings} shape {@code [typeVocabSize, H]}</li>
 * </ul>
 */
public final class BertEmbeddingLookup {

    private BertEmbeddingLookup() {
    }

    /**
     * Allocates a {@code [seqBucket * H]} float array and fills the
     * leading {@code seqLen} rows with the summed embeddings.
     *
     * @param cfg                 architecture knobs ({@code hiddenSize}).
     * @param wordEmbeddings      table of shape {@code [vocabSize, H]}.
     * @param positionEmbeddings  table of shape {@code [maxPositionEmbeddings, H]}.
     * @param tokenTypeEmbeddings table of shape {@code [typeVocabSize, H]}.
     * @param encoded             tokenizer output (provides {@code inputIds},
     *                            {@code tokenTypeIds}, {@code length}).
     * @param seqBucket           padded sequence length the GPU stack expects
     *                            ({@code >= encoded.length()}).
     * @return new {@code float[seqBucket * H]} with rows
     * {@code [encoded.length(), seqBucket)} left as zero.
     */
    public static float[] lookup(BertEncoderConfig cfg,
                                 float[] wordEmbeddings,
                                 float[] positionEmbeddings,
                                 float[] tokenTypeEmbeddings,
                                 EncoderTokenizer.Encoded encoded,
                                 int seqBucket) {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(wordEmbeddings, "wordEmbeddings");
        Objects.requireNonNull(positionEmbeddings, "positionEmbeddings");
        Objects.requireNonNull(tokenTypeEmbeddings, "tokenTypeEmbeddings");
        Objects.requireNonNull(encoded, "encoded");
        int seqLen = encoded.length();
        if (seqLen <= 0) {
            throw new IllegalArgumentException("encoded.length() must be > 0");
        }
        if (seqBucket < seqLen) {
            throw new IllegalArgumentException(
                    "seqBucket (" + seqBucket + ") < seqLen (" + seqLen + ")");
        }
        int H = cfg.hiddenSize();
        float[] x = new float[(long) seqBucket * H > Integer.MAX_VALUE
                ? (int) (Integer.MAX_VALUE) : seqBucket * H];
        int[] ids = encoded.inputIds();
        int[] tts = encoded.tokenTypeIds();
        for (int t = 0; t < seqLen; t++) {
            int id = ids[t];
            int tt = tts[t];
            int dst = t * H;
            int ws = id * H;
            int ps = t * H;
            int tos = tt * H;
            for (int h = 0; h < H; h++) {
                x[dst + h] = wordEmbeddings[ws + h]
                        + positionEmbeddings[ps + h]
                        + tokenTypeEmbeddings[tos + h];
            }
        }
        return x;
    }
}

