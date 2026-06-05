package com.aresstack.windirectml.inference.t5;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Selects the tokenizer implementation for a curated T5-family model directory.
 */
final class T5TokenizerLoader {
    private T5TokenizerLoader() {
    }

    static T5TextTokenizer load(Path modelDir) throws IOException {
        Objects.requireNonNull(modelDir, "modelDir");
        if (Files.isRegularFile(modelDir.resolve("vocab.json"))
                && Files.isRegularFile(modelDir.resolve("merges.txt"))) {
            return CodeT5Tokenizer.load(modelDir);
        }
        if (Files.isRegularFile(modelDir.resolve("tokenizer.json"))) {
            return JsonT5Tokenizer.load(modelDir);
        }
        throw new IOException("No supported T5 tokenizer found in " + modelDir
                + " (expected CodeT5 vocab.json+merges.txt or google-t5 tokenizer.json)");
    }

    static String describeMissingTokenizer(Path modelDir) {
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            return "T5 model directory not found: " + modelDir;
        }
        boolean hasCodeT5Tokenizer = Files.isRegularFile(modelDir.resolve("vocab.json"))
                && Files.isRegularFile(modelDir.resolve("merges.txt"));
        boolean hasTokenizerJson = Files.isRegularFile(modelDir.resolve("tokenizer.json"));
        if (hasCodeT5Tokenizer || hasTokenizerJson) {
            return null;
        }
        return "Missing T5 tokenizer files: expected vocab.json+merges.txt or tokenizer.json";
    }
}
