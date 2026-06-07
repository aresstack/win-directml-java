package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class CodeT5TokenizerTest {
    @TempDir
    Path tempDir;

    @Test
    void encodesAndDecodesByteLevelTokens() throws Exception {
        writeTokenizerFiles(tempDir);

        CodeT5Tokenizer tokenizer = CodeT5Tokenizer.load(tempDir);

        int[] encoded = tokenizer.encode("Hi");

        assertArrayEquals(new int[]{1, 4, 5, 2}, encoded);
        assertEquals("Hi", tokenizer.decode(encoded));
        assertEquals(1, tokenizer.bosTokenId());
        assertEquals(2, tokenizer.eosTokenId());
    }

    @Test
    void skipsSpecialTokensWhenDecoding() throws Exception {
        writeTokenizerFiles(tempDir);

        CodeT5Tokenizer tokenizer = CodeT5Tokenizer.load(tempDir);

        assertEquals("Hi", tokenizer.decode(new int[]{1, 4, 5, 2, 0}));
    }

    static void writeTokenizerFiles(Path modelDir) throws Exception {
        Files.createDirectories(modelDir);
        Files.writeString(modelDir.resolve("vocab.json"), "{\n" +
                "  \"<pad>\": 0,\n" +
                "  \"<s>\": 1,\n" +
                "  \"</s>\": 2,\n" +
                "  \"<unk>\": 3,\n" +
                "  \"H\": 4,\n" +
                "  \"i\": 5\n" +
                "}\n", StandardCharsets.UTF_8);
        Files.writeString(modelDir.resolve("merges.txt"), "#version: 0.2\n", StandardCharsets.UTF_8);
        Files.writeString(modelDir.resolve("special_tokens_map.json"), "{\n" +
                "  \"bos_token\": \"<s>\",\n" +
                "  \"eos_token\": \"</s>\",\n" +
                "  \"unk_token\": \"<unk>\",\n" +
                "  \"pad_token\": \"<pad>\"\n" +
                "}\n", StandardCharsets.UTF_8);
    }
}
