package com.aresstack.windirectml.inference.smollm2;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2TokenizerTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

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

    @Test
    void tokenizerUsesGpt2NumberPreTokenizationWithoutThreeDigitSplit() throws Exception {
        Path tokenizerJson = tempDir.resolve("gpt2-number-tokenizer.json");
        writeGpt2PreTokenizerFixture(tokenizerJson);
        SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerJson);

        assertArrayEquals(new int[]{9}, tokenizer.encode(" 1234"));
    }

    @Test
    void tokenizerDoesNotAttachPunctuationToFollowingWord() throws Exception {
        Path tokenizerJson = tempDir.resolve("gpt2-punctuation-tokenizer.json");
        writeGpt2PreTokenizerFixture(tokenizerJson);
        SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerJson);

        assertArrayEquals(new int[]{10, 15}, tokenizer.encode(".Hello"));
    }

    private static void writeGpt2PreTokenizerFixture(Path tokenizerJson) throws Exception {
        Map<String, Object> model = new LinkedHashMap<>();
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("Ġ", 0);
        vocab.put("1", 1);
        vocab.put("2", 2);
        vocab.put("3", 3);
        vocab.put("4", 4);
        vocab.put("H", 5);
        vocab.put("Ġ1", 6);
        vocab.put("Ġ12", 7);
        vocab.put("Ġ123", 8);
        vocab.put("Ġ1234", 9);
        vocab.put(".", 10);
        vocab.put("e", 11);
        vocab.put("l", 12);
        vocab.put("o", 13);
        vocab.put("He", 14);
        vocab.put("Hello", 15);
        model.put("type", "BPE");
        model.put("vocab", vocab);
        model.put("merges", List.of(
                List.of("Ġ", "1"),
                List.of("Ġ1", "2"),
                List.of("Ġ12", "3"),
                List.of("Ġ123", "4"),
                List.of("H", "e"),
                List.of("He", "l"),
                List.of("Hel", "l"),
                List.of("Hell", "o")
        ));
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("model", model);
        root.put("added_tokens", List.of());
        Files.writeString(tokenizerJson, MAPPER.writeValueAsString(root), StandardCharsets.UTF_8);
    }
}
