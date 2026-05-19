package com.aresstack.windirectml.encoder.tokenizer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the BERT NSP-style pair tokenization
 * ({@code [CLS] a [SEP] b [SEP]}, token_type_ids {@code [0…0,1…1]}).
 * Exercises the path required by the cross-encoder reranker.
 */
class WordPieceTokenizerPairTest {

    private static final String TOKENIZER_JSON = """
            {
              "version": "1.0",
              "normalizer": { "type": "BertNormalizer", "lowercase": true, "strip_accents": true },
              "pre_tokenizer": { "type": "BertPreTokenizer" },
              "added_tokens": [
                { "id": 0, "content": "[PAD]" },
                { "id": 1, "content": "[UNK]" },
                { "id": 2, "content": "[CLS]" },
                { "id": 3, "content": "[SEP]" }
              ],
              "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {
                  "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                  "hello": 10, "world": 11, "what": 12, "is": 13, "this": 14, "thing": 15
                }
              }
            }
            """;

    private static WordPieceTokenizer load(Path dir, int maxLen) throws Exception {
        Path file = dir.resolve("tokenizer.json");
        Files.writeString(file, TOKENIZER_JSON);
        return WordPieceTokenizer.load(file, maxLen);
    }

    @Test
    void encodePairProducesCanonicalShape(@TempDir Path dir) throws Exception {
        WordPieceTokenizer tok = load(dir, 64);
        var enc = tok.encodePair("what is this", "this thing");

        // [CLS] what is this [SEP] this thing [SEP]
        assertArrayEquals(new int[]{2, 12, 13, 14, 3, 14, 15, 3}, enc.inputIds());
        // token_type_ids: segment 0 covers CLS+A+first SEP, segment 1 covers B+second SEP.
        assertArrayEquals(new int[]{0, 0, 0, 0, 0, 1, 1, 1}, enc.tokenTypeIds());
        for (int m : enc.attentionMask()) assertEquals(1, m);
    }

    @Test
    void encodePairTruncatesLongestFirst(@TempDir Path dir) throws Exception {
        // Joint budget = maxLen - 3 (CLS + 2x SEP). With maxLen=6 → budget=3.
        WordPieceTokenizer tok = load(dir, 6);
        var enc = tok.encodePair("hello world hello", "this");
        // Expect the longer segment (A, 3 tokens) trimmed down so total = 3:
        //   A truncated to 2 tokens, B keeps its 1 token, total = 3 ≤ budget.
        assertTrue(enc.length() <= 6);
        // Last token must be [SEP] (id 3) – truncation never drops the closing marker.
        assertEquals(3, enc.inputIds()[enc.length() - 1]);
        // First token must remain [CLS] (id 2).
        assertEquals(2, enc.inputIds()[0]);
        // token_type_ids stay consistent with segment boundaries: zeros, then ones.
        int[] tt = enc.tokenTypeIds();
        boolean seenOne = false;
        for (int i = 0; i < tt.length; i++) {
            if (tt[i] == 1) seenOne = true;
            else if (seenOne) {
                throw new AssertionError("segment ids must be monotonically non-decreasing");
            }
        }
    }

    @Test
    void encodePairRejectsSingleSegmentTokenizers() {
        com.aresstack.windirectml.encoder.EncoderTokenizer noPair =
                new com.aresstack.windirectml.encoder.EncoderTokenizer() {
                    @Override public Encoded encode(String text) { return null; }
                    @Override public int padTokenId() { return 0; }
                    @Override public int clsTokenId() { return 0; }
                    @Override public int sepTokenId() { return 0; }
                    @Override public int vocabSize() { return 0; }
                };
        try {
            noPair.encodePair("a", "b");
            throw new AssertionError("expected UnsupportedOperationException");
        } catch (UnsupportedOperationException expected) {
            // good
        }
    }
}

