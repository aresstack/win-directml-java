package com.aresstack.windirectml.inference.whisper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Read the token suppression lists of a Hugging Face Whisper checkpoint. Many model
 * bundles ship without {@code generation_config.json}; for the multilingual vocabulary
 * the published reference lists are built in, so suppression never silently disappears.
 */
final class WhisperGenerationConfig {
    private static final int MULTILINGUAL_VOCABULARY = 51865;

    /** OpenAI's non-speech symbols plus task and prompt tokens for the multilingual vocabulary. */
    private static final int[] MULTILINGUAL_SUPPRESS_TOKENS = {
            1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93,
            359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246,
            3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929,
            11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362,
            18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
            36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362
    };
    private static final int[] MULTILINGUAL_BEGIN_SUPPRESS_TOKENS = {220, 50257};

    private final int[] suppressTokens;
    private final int[] beginSuppressTokens;

    private WhisperGenerationConfig(int[] suppressTokens, int[] beginSuppressTokens) {
        this.suppressTokens = suppressTokens;
        this.beginSuppressTokens = beginSuppressTokens;
    }

    static WhisperGenerationConfig load(Path modelDirectory, WhisperModelConfig config) throws IOException {
        Path file = modelDirectory.resolve("generation_config.json");
        boolean multilingual = config.vocabularySize() == MULTILINGUAL_VOCABULARY;
        int[] suppress = multilingual ? MULTILINGUAL_SUPPRESS_TOKENS.clone() : new int[0];
        int[] begin = multilingual ? MULTILINGUAL_BEGIN_SUPPRESS_TOKENS.clone() : new int[0];
        if (Files.isRegularFile(file)) {
            JsonNode root = new ObjectMapper().readTree(file.toFile());
            suppress = readTokens(root.path("suppress_tokens"), suppress);
            begin = readTokens(root.path("begin_suppress_tokens"), begin);
        }
        return new WhisperGenerationConfig(suppress, begin);
    }

    private static int[] readTokens(JsonNode node, int[] fallback) {
        if (!node.isArray()) {
            return fallback;
        }
        int[] tokens = new int[node.size()];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = node.get(i).asInt();
        }
        return tokens;
    }

    int[] suppressTokens() {
        return suppressTokens.clone();
    }

    int[] beginSuppressTokens() {
        return beginSuppressTokens.clone();
    }
}
