package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2TokenizerTest {

    @TempDir
    Path tempDir;

    @Test
    void tokenizerLoadsByteLevelBpeVocabulary() throws Exception {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        SmolLM2TestFixtures.writeTokenizerJson(tokenizerJson);

        SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerJson);

        assertArrayEquals(new int[]{2}, tokenizer.encode("ab"));
        assertEquals("ab", tokenizer.decode(new int[]{2}));
        assertEquals(4, tokenizer.vocabSize());
    }

    @Test
    void tokenizerPreservesSpecialTokensDuringEncoding() throws Exception {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        SmolLM2TestFixtures.writeTokenizerJson(tokenizerJson);

        SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerJson);

        assertArrayEquals(new int[]{0, 3, 1}, tokenizer.encode("a<|endoftext|>b"));
        assertTrue(tokenizer.isSpecialToken(3));
        assertEquals(Map.of("<|endoftext|>", 3), tokenizer.specialTokens());
    }

    @Test
    void tokenizerSkipsSpecialTokensDuringDecoding() throws Exception {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        SmolLM2TestFixtures.writeTokenizerJson(tokenizerJson);

        SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerJson);

        assertEquals("ab", tokenizer.decode(new int[]{0, 3, 1}, true));
    }
}
