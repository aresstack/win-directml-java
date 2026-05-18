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
        public int length() { return inputIds.length; }
    }
}

