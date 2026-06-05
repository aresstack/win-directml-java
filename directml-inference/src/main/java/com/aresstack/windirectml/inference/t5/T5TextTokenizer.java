package com.aresstack.windirectml.inference.t5;

/**
 * Minimal text/token bridge used by the T5 workbench runtime.
 *
 * <p>The inference runtime works on token ids. Implementations keep model-specific
 * tokenizer details out of {@link T5InferenceEngine}.</p>
 */
interface T5TextTokenizer {
    int[] encode(String text);

    String decode(int[] tokenIds);

    int vocabSize();
}
