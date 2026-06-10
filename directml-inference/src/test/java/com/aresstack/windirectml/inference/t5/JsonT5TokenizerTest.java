package com.aresstack.windirectml.inference.t5;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class JsonT5TokenizerTest {
    @TempDir
    Path tempDir;

    @Test
    void loadsGoogleT5StyleTokenizerJson() throws Exception {
        writeTokenizerFiles(tempDir);

        T5TextTokenizer tokenizer = T5TokenizerLoader.load(tempDir);

        int[] encoded = tokenizer.encode("hello world");

        assertArrayEquals(new int[]{3, 4, 1}, encoded);
        assertEquals("hello world", tokenizer.decode(encoded));
    }

    @Test
    void usesUnigramScoresInsteadOfGreedyLongestMatch() throws Exception {
        writeTokenizerFiles(tempDir);

        T5TextTokenizer tokenizer = T5TokenizerLoader.load(tempDir);

        int[] encoded = tokenizer.encode("helloed");

        assertArrayEquals(new int[]{3, 5, 1}, encoded);
        assertEquals("helloed", tokenizer.decode(encoded));
    }

    @Test
    void preservesSentencePieceWordBoundariesForPromptText() throws Exception {
        writeTokenizerFiles(tempDir);

        T5TextTokenizer tokenizer = T5TokenizerLoader.load(tempDir);

        int[] encoded = tokenizer.encode("summarize: Paste a longer text");

        assertEquals(1, encoded[encoded.length - 1]);
        assertEquals("summarize: Paste a longer text", tokenizer.decode(encoded));
    }



    @Test
    void skipsExtraIdTokensDuringVisibleDecode() throws Exception {
        writeTokenizerFiles(tempDir);

        T5TextTokenizer tokenizer = T5TokenizerLoader.load(tempDir);

        assertEquals("hello world", tokenizer.decode(new int[]{13, 3, 4, 1}));
    }
    static void writeTokenizerFiles(Path modelDir) throws Exception {
        Files.createDirectories(modelDir);
        Files.writeString(modelDir.resolve("tokenizer.json"), "{\n" +
                "  \"model\": {\n" +
                "    \"type\": \"Unigram\",\n" +
                "    \"vocab\": [\n" +
                "      [\"<pad>\", 0.0],\n" +
                "      [\"</s>\", 0.0],\n" +
                "      [\"<unk>\", -99.0],\n" +
                "      [\"▁hello\", -1.0],\n" +
                "      [\"▁world\", -1.0],\n" +
                "      [\"ed\", -1.0],\n" +
                "      [\"▁summarize\", -1.0],\n" +
                "      [\":\", -0.2],\n" +
                "      [\"▁Paste\", -1.0],\n" +
                "      [\"▁a\", -1.0],\n" +
                "      [\"▁longer\", -1.0],\n" +
                "      [\"▁text\", -1.0],\n" +
                "      [\"▁helloed\", -9.0],\n" +
                "      [\"<extra_id_0>\", 0.0]\n" +
                "    ]\n" +
                "  }\n" +
                "}\n", StandardCharsets.UTF_8);
        Files.writeString(modelDir.resolve("spiece.model"), "placeholder", StandardCharsets.UTF_8);
    }
}
