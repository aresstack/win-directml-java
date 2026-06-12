package com.aresstack.windirectml.inference.decoderonly;

/**
 * Next-token selection seam for the shared {@link DecoderOnlyGenerationLoop}.
 *
 * <p>Decouples the loop from any family-specific sampling/penalty policy: greedy, temperature/top-k/top-p sampling and
 * repetition penalties all live behind this interface. A model family supplies a concrete selector per generation
 * request (built from its own generation options + stop policy).</p>
 */
public interface DecoderOnlyTokenSelector {

    /**
     * Choose the next token id from the current logits, given the tokens generated so far (for repetition penalties).
     */
    int selectNextToken(float[] logits, DecoderOnlyGeneratedTokens generatedTokens);

    /** Whether {@code tokenId} ends generation (e.g. an EOS/stop token). */
    boolean shouldStop(int tokenId);
}
