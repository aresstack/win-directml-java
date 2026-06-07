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

    default String describeToken(int tokenId) {
        return String.valueOf(tokenId);
    }

    default String describeTokens(int[] tokenIds, int limit) {
        if (tokenIds == null || tokenIds.length == 0 || limit <= 0) {
            return "[]";
        }
        StringBuilder builder = new StringBuilder();
        builder.append('[');
        int count = Math.min(tokenIds.length, limit);
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                builder.append(", ");
            }
            builder.append(describeToken(tokenIds[i]));
        }
        if (count < tokenIds.length) {
            builder.append(", ...");
        }
        builder.append(']');
        return builder.toString();
    }
}
