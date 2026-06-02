package com.aresstack.windirectml.encoder.tokenizer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WordPieceTokenizerTest {

    /**
     * Minimaler BERT-uncased Tokenizer mit handpicked Vocab.
     */
    private static final String TOKENIZER_JSON = """
            {
              "version": "1.0",
              "normalizer": { "type": "BertNormalizer", "lowercase": true, "strip_accents": true },
              "pre_tokenizer": { "type": "BertPreTokenizer" },
              "added_tokens": [
                { "id": 0, "content": "[PAD]"  },
                { "id": 1, "content": "[UNK]"  },
                { "id": 2, "content": "[CLS]"  },
                { "id": 3, "content": "[SEP]"  }
              ],
              "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {
                  "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                  "hello": 10, "world": 11, "!": 12,
                  "to": 20, "##ken": 21, "##izer": 22,
                  ".": 30, "ai": 31
                }
              }
            }
            """;

    @Test
    void encodesSimpleSentence(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("tokenizer.json");
        Files.writeString(file, TOKENIZER_JSON);

        WordPieceTokenizer tok = WordPieceTokenizer.load(file, 64);

        var encoded = tok.encode("Hello world!");
        // [CLS] hello world ! [SEP]
        assertArrayEquals(new int[]{2, 10, 11, 12, 3}, encoded.inputIds());
        assertArrayEquals(new int[]{1, 1, 1, 1, 1}, encoded.attentionMask());
        assertEquals(encoded.inputIds().length, encoded.attentionMask().length);
    }

    @Test
    void splitsWordPiecesWithContinuingPrefix(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("tokenizer.json");
        Files.writeString(file, TOKENIZER_JSON);
        WordPieceTokenizer tok = WordPieceTokenizer.load(file, 64);

        var encoded = tok.encode("tokenizer");
        // [CLS] to ##ken ##izer [SEP]
        assertArrayEquals(new int[]{2, 20, 21, 22, 3}, encoded.inputIds());
    }

    @Test
    void stripsAccentsAndLowercases(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("tokenizer.json");
        Files.writeString(file, TOKENIZER_JSON);
        WordPieceTokenizer tok = WordPieceTokenizer.load(file, 64);

        var encoded = tok.encode("HÉLLO");
        // hello after normalize → token id 10
        assertTrue(contains(encoded.inputIds(), 10));
    }

    @Test
    void truncatesToMaxSequenceLength(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("tokenizer.json");
        Files.writeString(file, TOKENIZER_JSON);
        WordPieceTokenizer tok = WordPieceTokenizer.load(file, 4);

        var encoded = tok.encode("hello world hello world hello");
        assertTrue(encoded.inputIds().length <= 4);
        assertEquals(tok.sepTokenId(), encoded.inputIds()[encoded.inputIds().length - 1]);
    }

    private static boolean contains(int[] arr, int v) {
        for (int x : arr) if (x == v) return true;
        return false;
    }
}

