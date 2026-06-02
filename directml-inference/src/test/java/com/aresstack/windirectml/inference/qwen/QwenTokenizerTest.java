package com.aresstack.windirectml.inference.qwen;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link QwenTokenizer}.
 *
 * <p>Uses a minimal synthetic tokenizer.json that exercises the byte-level BPE
 * logic with a small vocabulary. The vocab includes:
 * <ul>
 *   <li>All 256 byte-level base characters (GPT-2 byte-to-unicode mapping)</li>
 *   <li>Common merged tokens: "he", "ll", "lo", "Hello", " world" etc.</li>
 *   <li>German characters via byte sequences (ä = 0xC3 0xA4)</li>
 *   <li>Code tokens: "def", "return", "(", ")", ":" etc.</li>
 *   <li>Special tokens: {@code <|endoftext|>}, {@code <|im_start|>}, {@code <|im_end|>}</li>
 * </ul>
 *
 * <p>Tests verify:
 * <ul>
 *   <li>Encode/decode round-trip for English text</li>
 *   <li>Encode/decode round-trip for German text with umlauts</li>
 *   <li>Encode/decode round-trip for code snippets</li>
 *   <li>Special token handling</li>
 *   <li>Empty/null input handling</li>
 * </ul>
 */
class QwenTokenizerTest {

    private static QwenTokenizer tokenizer;

    @TempDir
    static Path tempDir;

    @BeforeAll
    static void loadTokenizer() throws IOException {
        Path tokenizerJson = tempDir.resolve("tokenizer.json");
        Files.writeString(tokenizerJson, buildTestTokenizerJson());
        tokenizer = QwenTokenizer.load(tokenizerJson);
    }

    // ── Basic tests ──────────────────────────────────────────────────────

    @Test
    void loadsWithoutCloudCalls() {
        assertNotNull(tokenizer);
        assertTrue(tokenizer.vocabSize() > 0);
    }

    @Test
    void encodesEmptyString() {
        assertArrayEquals(new int[0], tokenizer.encode(""));
        assertArrayEquals(new int[0], tokenizer.encode(null));
    }

    // ── English round-trip ───────────────────────────────────────────────

    @Test
    void encodeDecodeEnglish() {
        String input = "Hello world";
        int[] ids = tokenizer.encode(input);
        assertTrue(ids.length > 0, "Encoding should produce tokens");
        String decoded = tokenizer.decode(ids);
        assertEquals(input, decoded);
    }

    @Test
    void encodeDecodeEnglishSentence() {
        String input = "The quick brown fox.";
        int[] ids = tokenizer.encode(input);
        assertTrue(ids.length > 0);
        String decoded = tokenizer.decode(ids);
        assertEquals(input, decoded);
    }

    // ── German round-trip (umlauts, punctuation) ─────────────────────────

    @Test
    void encodeDecodeGerman() {
        String input = "Hallo Welt";
        int[] ids = tokenizer.encode(input);
        assertTrue(ids.length > 0);
        String decoded = tokenizer.decode(ids);
        assertEquals(input, decoded);
    }

    @Test
    void encodeDecodeGermanUmlauts() {
        // ä ö ü ß are multi-byte UTF-8 characters
        String input = "Gr\u00fc\u00dfe";  // "Grüße"
        int[] ids = tokenizer.encode(input);
        assertTrue(ids.length > 0);
        String decoded = tokenizer.decode(ids);
        assertEquals(input, decoded);
    }

    // ── Code round-trip ──────────────────────────────────────────────────

    @Test
    void encodeDecodeCode() {
        String input = "def foo():";
        int[] ids = tokenizer.encode(input);
        assertTrue(ids.length > 0);
        String decoded = tokenizer.decode(ids);
        assertEquals(input, decoded);
    }

    @Test
    void encodeDecodeCodeWithSymbols() {
        String input = "x = 1 + 2";
        int[] ids = tokenizer.encode(input);
        assertTrue(ids.length > 0);
        String decoded = tokenizer.decode(ids);
        assertEquals(input, decoded);
    }

    // ── Special tokens ───────────────────────────────────────────────────

    @Test
    void specialTokensAreRecognized() {
        assertTrue(tokenizer.specialTokensMap().containsKey("<|endoftext|>"));
        assertTrue(tokenizer.specialTokensMap().containsKey("<|im_start|>"));
        assertTrue(tokenizer.specialTokensMap().containsKey("<|im_end|>"));
    }

    @Test
    void encodesSpecialTokensAsIds() {
        int[] ids = tokenizer.encode("<|im_start|>system");
        // First token should be the im_start special token
        assertEquals(QwenTokenizer.IM_START_ID, ids[0]);
    }

    @Test
    void decodePreservesSpecialTokensByDefault() {
        String input = "<|im_start|>user\nHello<|im_end|>\n";
        int[] ids = tokenizer.encode(input);
        String decoded = tokenizer.decode(ids);
        assertEquals(input, decoded);
    }

    @Test
    void decodeCanSkipSpecialTokensWhenRequested() {
        String input = "<|im_start|>user\nHello<|im_end|>\n";
        int[] ids = tokenizer.encode(input);
        String decoded = tokenizer.decode(ids, true);
        assertFalse(decoded.contains("<|im_start|>"));
        assertFalse(decoded.contains("<|im_end|>"));
    }

    @Test
    void isEosRecognizesStopTokens() {
        assertTrue(tokenizer.isEos(QwenTokenizer.ENDOFTEXT_ID));
        assertTrue(tokenizer.isEos(QwenTokenizer.IM_END_ID));
        assertFalse(tokenizer.isEos(0));
        assertFalse(tokenizer.isEos(100));
    }

    // ── Test tokenizer.json builder ──────────────────────────────────────

    /**
     * Build a minimal Qwen-format tokenizer.json for testing.
     * Uses byte-level BPE with a small set of merges.
     */
    private static String buildTestTokenizerJson() {
        StringBuilder vocab = new StringBuilder();
        // Add all 256 byte-level base characters
        char[] byteToUnicode = new char[256];
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
                byteToUnicode[b] = (char) b;
            } else {
                byteToUnicode[b] = (char) (256 + n);
                n++;
            }
        }

        for (int b = 0; b < 256; b++) {
            String key = jsonEscape(String.valueOf(byteToUnicode[b]));
            if (vocab.length() > 0) vocab.append(",\n");
            vocab.append("    \"").append(key).append("\": ").append(b);
        }

        // Add merged tokens (IDs start at 256)
        String[][] mergedTokens = {
                {"He", "256"}, {"ll", "257"}, {"lo", "258"},
                {"Hel", "259"}, {"Hell", "260"}, {"Hello", "261"},
                {"Ġw", "262"}, {"or", "263"}, {"ld", "264"},
                {"orl", "265"}, {"orld", "266"}, {"Ġworld", "267"},
                {"Ha", "268"}, {"Hal", "269"}, {"Hall", "270"}, {"Hallo", "271"},
                {"Ġ", "272"}, {"We", "273"}, {"lt", "274"},
                {"Wel", "275"}, {"Welt", "276"},
                {"de", "277"}, {"ef", "278"}, {"def", "279"},
                {"fo", "280"}, {"oo", "281"}, {"foo", "282"},
                {"Th", "283"}, {"The", "284"},
                {"qu", "285"}, {"ic", "286"}, {"ck", "287"},
                {"ick", "288"}, {"uick", "289"}, {"quick", "290"},
                {"br", "291"}, {"ow", "292"}, {"own", "293"}, {"rown", "294"}, {"brown", "295"},
                {"Ġf", "296"}, {"ox", "297"},
                {"Ġfo", "298"}, {"Ġfox", "299"},
                {"re", "300"}, {"tu", "301"}, {"rn", "302"},
                {"tur", "303"}, {"turn", "304"}, {"eturn", "305"}, {"return", "306"},
        };

        for (String[] entry : mergedTokens) {
            String key = entry[0]
                    .replace("Ġ", String.valueOf(byteToUnicode[' ']));
            vocab.append(",\n    \"").append(jsonEscape(key)).append("\": ").append(entry[1]);
        }

        // Merges (ordered by priority)
        String[] merges = {
                "H e", "l l", "l o",
                "He l", "Hel l", "Hell o",
                String.valueOf(byteToUnicode[' ']) + " w",  // Ġw merge – not needed as merge string
                "o r", "l d",
                "or l", "orl d",
                // We'll express merges in the byte-unicode form
        };

        // Build merges in proper format
        String spChar = String.valueOf(byteToUnicode[' ']);
        String[] allMerges = {
                "H e", "l l", "l o",
                "He l", "Hel l", "Hell o",
                spChar + " w", "o r", "l d",
                "or l", "orl d", spChar + "w orld",
                "H a", "Ha l", "Hal l", "Hall o",
                spChar + " W", "W e", "l t",
                "We l", "Wel t",
                "d e", "e f", "de f",
                "f o", "o o", "fo o",
                "T h", "Th e",
                "q u", "i c", "c k",
                "ic k", "u ick", "qu ick",  // "quick" needs work
                "b r", "o w", "ow n", "r own", "br own",
                spChar + " f", "o x",
                spChar + "f o", spChar + "fo x",
                "r e", "t u", "r n",
                "tu r", "tur n", "e turn", "re turn",
        };

        StringBuilder mergesJson = new StringBuilder();
        for (int i = 0; i < allMerges.length; i++) {
            if (i > 0) mergesJson.append(",\n");
            mergesJson.append("    \"").append(jsonEscape(allMerges[i])).append("\"");
        }

        // Special tokens
        int endoftextId = 151643;
        int imStartId = 151644;
        int imEndId = 151645;

        return """
                {
                  "model": {
                    "type": "BPE",
                    "vocab": {
                %s
                    },
                    "merges": [
                %s
                    ]
                  },
                  "added_tokens": [
                    {"id": %d, "content": "<|endoftext|>", "special": true},
                    {"id": %d, "content": "<|im_start|>", "special": true},
                    {"id": %d, "content": "<|im_end|>", "special": true}
                  ]
                }
                """.formatted(vocab, mergesJson, endoftextId, imStartId, imEndId);
    }

    private static String jsonEscape(String s) {
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == '"') sb.append("\\\"");
            else if (c == '\\') sb.append("\\\\");
            else if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
            else sb.append(c);
        }
        return sb.toString();
    }
}
