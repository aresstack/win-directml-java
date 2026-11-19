package com.aresstack.windirectml.inference.whisper;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Compare the Java BPE encoder with IDs produced by the Hugging Face tokenizers library. */
class WhisperTokenizerEncodeTest {
    private static final Path MODEL = Path.of("..", "model", "whisper-tiny");

    @Test
    void encodesTextExactlyLikeHuggingFace() throws Exception {
        Assumptions.assumeTrue(Files.isRegularFile(MODEL.resolve("vocab.json"))
                && Files.isRegularFile(MODEL.resolve("tokenizer.json")));
        WhisperTokenizer tokenizer = new WhisperTokenizer(MODEL);

        assertEquals(Arrays.asList(21242, 14761, 13), tokenizer.encode(" Hallo Welt."));
        assertEquals(Arrays.asList(53, 284, 2265, 260, 9394, 17111, 1147, 11078, 1343, 29077, 328, 674, 3018,
                29077, 328, 259, 13), tokenizer.encode("Vor langer Zeit lebten einmal ein König und eine Königin."));
        assertEquals(Arrays.asList(45778, 11451, 25, 2272, 11, 20, 14189, 1662, 1059, 252, 57, 14032, 913, 250,
                5799, 3133, 30), tokenizer.encode(" Größe: 12,5 % – „Zitat“ … ok?"));
        assertEquals(Arrays.asList(8705, 3653, 85, 3011, 11, 11701, 40432, 12, 54, 891, 694, 32743, 311, 220, 220,
                45200, 198, 2009, 10880), tokenizer.encode(" Space Harvester, Confluence-Wissensquelle's   Seiten\n2026"));
    }

    @Test
    void decodesEncodedTextBackToTheOriginal() throws Exception {
        Assumptions.assumeTrue(Files.isRegularFile(MODEL.resolve("vocab.json")));
        WhisperTokenizer tokenizer = new WhisperTokenizer(MODEL);
        String text = " Die Königstochter soll sich an ihrem fünfzehnten Geburtstag an einer Spindel stechen!";

        assertEquals(text, tokenizer.decode(tokenizer.encode(text)));
    }

    @Test
    void listsTheLanguageTokensOfTheMultilingualCheckpoint() throws Exception {
        Assumptions.assumeTrue(Files.isRegularFile(MODEL.resolve("vocab.json")));
        java.util.Map<String, Integer> languages = new WhisperTokenizer(MODEL).languageTokens();

        assertEquals(99, languages.size());
        assertEquals(50259, languages.get("en"));
        assertEquals(50261, languages.get("de"));
        assertEquals(50357, languages.get("su"));
    }
}
