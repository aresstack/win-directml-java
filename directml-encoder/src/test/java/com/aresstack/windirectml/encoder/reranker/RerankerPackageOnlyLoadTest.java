package com.aresstack.windirectml.encoder.reranker;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.pack.EncoderWdmlPack;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Device-free, model-free proof of the P3 fix: the reranker completeness check
 * requires the runtime package ({@code reranker.wdmlpack}) — the artefact that
 * {@link RerankerCpuWeights#load} actually consumes — and no longer the raw
 * {@code model.safetensors}.
 *
 * <p>These tests exercise {@code BertCrossEncoderRerankers.verifyDir} indirectly
 * through {@code loadCpu}: they assert only the <em>completeness</em> outcome
 * (whether the directory is rejected as "incomplete"), not weight parsing, so
 * they need neither real weights nor a GPU.
 */
class RerankerPackageOnlyLoadTest {

    private static void write(Path file, String content) throws Exception {
        Files.write(file, content.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    /**
     * A package-only directory (reranker.wdmlpack + tokenizer.json + config.json,
     * NO model.safetensors) passes the completeness check. The load still fails
     * afterwards — the stub package is not real weights — but with a
     * package-parse error, never the "incomplete directory" error.
     */
    @Test
    void packageOnlyDirectoryPassesCompletenessCheck(@TempDir Path dir) throws Exception {
        write(dir.resolve(EncoderWdmlPack.RERANKER_PACKAGE_FILE), "not-a-real-package");
        write(dir.resolve("tokenizer.json"), "{}");
        write(dir.resolve("config.json"),
                "{\"hidden_size\":8,\"num_hidden_layers\":1,\"num_attention_heads\":2,"
                        + "\"intermediate_size\":16,\"max_position_embeddings\":16,\"vocab_size\":32,"
                        + "\"type_vocab_size\":2,\"layer_norm_eps\":1e-12,\"hidden_act\":\"gelu\"}");
        assertTrue(Files.notExists(dir.resolve("model.safetensors")),
                "precondition: no raw weights present");

        EmbeddingException ex =
                assertThrows(EmbeddingException.class, () -> BertCrossEncoderRerankers.loadCpu(dir));
        String msg = ex.getMessage() == null ? "" : ex.getMessage().toLowerCase(Locale.ROOT);
        assertTrue(!msg.contains("incomplete"),
                "package-only directory must clear the completeness check, got: " + ex.getMessage());
    }

    /**
     * A directory with only the raw weights (model.safetensors + tokenizer.json +
     * config.json) but NO reranker.wdmlpack now fails closed as "incomplete",
     * pointing at the missing runtime package — the raw weights are never read.
     */
    @Test
    void rawWeightsWithoutPackageFailIncomplete(@TempDir Path dir) throws Exception {
        write(dir.resolve("model.safetensors"), "raw");
        write(dir.resolve("tokenizer.json"), "{}");
        write(dir.resolve("config.json"), "{}");
        assertTrue(Files.notExists(dir.resolve(EncoderWdmlPack.RERANKER_PACKAGE_FILE)),
                "precondition: no runtime package present");

        EmbeddingException ex =
                assertThrows(EmbeddingException.class, () -> BertCrossEncoderRerankers.loadCpu(dir));
        String msg = ex.getMessage() == null ? "" : ex.getMessage().toLowerCase(Locale.ROOT);
        assertTrue(msg.contains("incomplete"),
                "raw-weights-only directory must fail the completeness check, got: " + ex.getMessage());
        assertTrue(msg.contains(EncoderWdmlPack.RERANKER_PACKAGE_FILE),
                "message should name the missing runtime package, got: " + ex.getMessage());
    }
}
