package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * GEMMA-WARP-2: the native Java tokenizer must match HF transformers exactly. Expected ids were
 * captured from {@code AutoTokenizer("google/gemma-3-270m-it")} (each result includes the prepended
 * {@code <bos>}=2). Gated on the local tokenizer.json being present; no Python at test time.
 */
@EnabledIf("tokenizerPresent")
class Gemma3TokenizerParityTest {

    static Path resolveTokenizerJson() {
        String override = System.getProperty("gemma.testModelDir");
        if (override != null && !override.isBlank()) {
            return fileIfExists(Path.of(override, "tokenizer.json"));
        }
        String appData = System.getenv("APPDATA");
        if (appData != null && !appData.isBlank()) {
            Path p = fileIfExists(Path.of(appData, ".directml", "model", "gemma-3-270m-it", "tokenizer.json"));
            if (p != null) {
                return p;
            }
        }
        String home = System.getProperty("user.home");
        return home == null ? null
                : fileIfExists(Path.of(home, ".directml", "model", "gemma-3-270m-it", "tokenizer.json"));
    }

    private static Path fileIfExists(Path p) {
        return p != null && Files.isRegularFile(p) ? p : null;
    }

    static boolean tokenizerPresent() {
        return resolveTokenizerJson() != null;
    }

    private static Gemma3Tokenizer tokenizer() throws Exception {
        return Gemma3Tokenizer.load(resolveTokenizerJson());
    }

    @Test
    void matchesTransformersForFixtures() throws Exception {
        Gemma3Tokenizer tok = tokenizer();
        assertArrayEquals(new int[]{2, 818, 5279, 529, 7001, 563, 9079, 236761},
                tok.encode("The capital of France is Paris."), "english");
        assertArrayEquals(new int[]{2, 9322, 237001, 16756, 941, 83754, 236787, 31083, 22744, 535,
                        236764, 110058, 236764, 15356, 51554, 236761},
                tok.encode("Größe und Maß: Äpfel, Öl, Übung."), "german umlauts");
        assertArrayEquals(new int[]{2, 11913, 997, 178318, 236788, 1740, 13222, 37440},
                tok.encode("READ #EMPLOYEES BY NAME"), "natural/adabas");
        assertArrayEquals(new int[]{2, 8344, 3805, 566, 578, 106703, 236761, 22331, 885, 2202, 1443},
                tok.encode("float[] v = embeddings.embed(\"hi\");"), "java");
        assertArrayEquals(new int[]{2}, tok.encode(""), "empty -> just BOS");
        assertArrayEquals(new int[]{2, 67906, 163543, 532, 236743, 201637},
                tok.encode("emoji 😀 and 漢字"), "unicode/byte-fallback");
    }

    @Test
    void chatTemplateMatchesTransformers() throws Exception {
        Gemma3Tokenizer tok = tokenizer();
        String prompt = Gemma3ChatTemplate.renderUserTurn("Hello");
        assertArrayEquals(new int[]{2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107},
                tok.encode(prompt), "chat user turn");
    }

    @Test
    void encodeDecodeRoundTripIsReadable() throws Exception {
        Gemma3Tokenizer tok = tokenizer();
        String text = "The capital of France is Paris.";
        assertEquals(text, tok.decode(tok.encode(text)).strip());
    }
}
