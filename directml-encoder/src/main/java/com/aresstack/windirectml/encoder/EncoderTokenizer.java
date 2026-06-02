package com.aresstack.windirectml.encoder;

/**
 * Tokenizer abstraction for encoder models.
 * <p>
 * Encoder tokenizers (WordPiece, SentencePiece-BPE, …) differ from
 * Phi-3's tokenizer and must not share its implementation. The encoder
 * runtime consumes {@link Encoded} structs only.
 */
public interface EncoderTokenizer {

    Encoded encode(String text);

    /**
     * Cross-encoder / NSP-style pair encoding:
     * {@code [CLS] textA [SEP] textB [SEP]} with {@code token_type_ids}
     * {@code [0, 0, …, 0, 1, 1, …, 1]} – segment 0 for the {@code [CLS]}
     * and the first half (incl. its trailing {@code [SEP]}), segment 1
     * for the second half (incl. its trailing {@code [SEP]}).
     * <p>
     * The default throws {@link UnsupportedOperationException} so single-
     * segment tokenizers (SentencePiece-only families) are not forced to
     * implement it.
     */
    default Encoded encodePair(String textA, String textB) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not implement encodePair");
    }

    int padTokenId();

    int clsTokenId();

    int sepTokenId();

    int vocabSize();

    /**
     * Encoder input tensors.
     *
     * @param inputIds      token ids of shape {@code [seq_len]}.
     * @param attentionMask 1 for real tokens, 0 for padding.
     * @param tokenTypeIds  segment ids; {@code null} for single-segment models.
     */
    record Encoded(int[] inputIds, int[] attentionMask, int[] tokenTypeIds) {
        public int length() {
            return inputIds.length;
        }
    }
}

