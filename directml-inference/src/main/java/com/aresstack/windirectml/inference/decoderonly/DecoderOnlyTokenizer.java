package com.aresstack.windirectml.inference.decoderonly;

/**
 * Text ↔ token-id seam for decoder-only model families.
 *
 * <p>The adapter point a decoder-only family's tokenizer implements so higher layers (runtime, generation) can encode
 * prompts and decode generated ids without depending on a concrete family tokenizer. SmolLM2 is the first consumer;
 * Qwen plugs into the same seam later. Chat-template rendering stays out of the tokenizer (see the prompt pipeline's
 * {@code PromptStrategy}).</p>
 */
public interface DecoderOnlyTokenizer {

    /** Encode {@code text} into token ids. */
    int[] encode(String text);

    /** Decode token {@code ids} back into text. */
    String decode(int[] ids);

    /** Decode token {@code ids} back into text, optionally skipping special tokens. */
    String decode(int[] ids, boolean skipSpecialTokens);

    /** Decode a single token id to its surface text. */
    String decodeToken(int id);

    /** Vocabulary size. */
    int vocabSize();

    /** Whether {@code tokenId} is a special (non-surface) token. */
    boolean isSpecialToken(int tokenId);
}
