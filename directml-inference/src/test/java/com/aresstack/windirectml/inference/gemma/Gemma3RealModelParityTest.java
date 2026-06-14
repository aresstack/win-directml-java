package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Real-model parity for the Gemma 3 CPU reference (GEMMA-WARP-9 validation): the Java reference forward
 * pass must reproduce transformers' greedy next token. Gated on the local model being present (override
 * with {@code -Dgemma.testModelDir=...}); skipped in CI.
 *
 * <p>Reference captured from HF transformers (torch.float32): for prompt "The capital of France is"
 * the input ids are {@code [2,818,5279,529,7001,563]} and argmax(next) = {@code 9079} (" Paris").</p>
 */
@EnabledIf("modelPresent")
class Gemma3RealModelParityTest {

    private static final int[] FRANCE_IDS = {2, 818, 5279, 529, 7001, 563};
    private static final int EXPECTED_NEXT = 9079; // " Paris" per transformers

    static Path resolveModelDir() {
        String override = System.getProperty("gemma.testModelDir");
        if (override != null && !override.isBlank()) {
            return dirIfValid(Path.of(override));
        }
        String appData = System.getenv("APPDATA");
        if (appData != null && !appData.isBlank()) {
            Path p = dirIfValid(Path.of(appData, ".directml", "model", "gemma-3-270m-it"));
            if (p != null) {
                return p;
            }
        }
        String home = System.getProperty("user.home");
        if (home != null) {
            return dirIfValid(Path.of(home, ".directml", "model", "gemma-3-270m-it"));
        }
        return null;
    }

    private static Path dirIfValid(Path dir) {
        return dir != null && Files.isRegularFile(dir.resolve("config.json"))
                && Files.isRegularFile(dir.resolve("model.safetensors")) ? dir : null;
    }

    static boolean modelPresent() {
        return resolveModelDir() != null;
    }

    @Test
    void referenceForwardMatchesTransformersGreedyNextToken() throws Exception {
        Path dir = resolveModelDir();
        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        assertEquals(262144, config.vocabSize(), "expected the real 270M-it config");

        SafeTensorsReader.SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));
        Gemma3ReferenceWeights weights = Gemma3ReferenceWeights.load(file, config);
        Gemma3ReferenceForwardPass fp = new Gemma3ReferenceForwardPass(weights);

        float[] logits = fp.logitsForLastToken(FRANCE_IDS);
        int next = com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath.argmax(logits);
        System.out.println("[GEMMA-PARITY] next token id = " + next + " (expected " + EXPECTED_NEXT + ")");
        assertTrue(Float.isFinite(logits[next]));
        assertEquals(EXPECTED_NEXT, next,
                "Java reference greedy next token must match transformers (\" Paris\")");
    }

    @Test
    void nativeTextToTextProducesParis() throws Exception {
        Path dir = resolveModelDir();
        Gemma3Config config = new Gemma3ConfigReader().read(dir.resolve("config.json"));
        Gemma3Tokenizer tokenizer = Gemma3Tokenizer.load(dir.resolve("tokenizer.json"));
        SafeTensorsReader.SafeTensorsFile file = SafeTensorsReader.read(dir.resolve("model.safetensors"));
        Gemma3ReferenceForwardPass fp = new Gemma3ReferenceForwardPass(
                Gemma3ReferenceWeights.load(file, config));

        // Full native chain: text -> Java tokenizer -> Java forward -> Java decode. No Python, no hardcoded ids.
        int[] ids = tokenizer.encode("The capital of France is");
        int next = com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath.argmax(fp.logitsForLastToken(ids));
        String decoded = tokenizer.decode(new int[]{next}).strip();
        System.out.println("[GEMMA-NATIVE] '" + decoded + "' (id " + next + ")");
        assertTrue(decoded.equals("Paris"), "native text->token->forward->text must yield 'Paris', got '" + decoded + "'");
    }
}
