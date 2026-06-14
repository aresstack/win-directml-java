package com.aresstack.windirectml.inference.gemma;

/**
 * Gemma 3 special token ids. Gemma uses {@code <bos>=2}, {@code <eos>=1}, {@code <pad>=0}, plus the
 * chat-turn tokens {@code <start_of_turn>} / {@code <end_of_turn>} (resolved by the tokenizer).
 */
public record Gemma3SpecialTokens(int bosTokenId, int eosTokenId, int padTokenId) {
}
