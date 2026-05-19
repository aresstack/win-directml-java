package com.aresstack.windirectml.encoder.e5;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.aresstack.windirectml.encoder.bert.BertConfigJson;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Structural tests for the variant parser, the {@code config.json}
 * reader and {@link BertConfigJson#verifyMatches} – no GPU and no
 * model files needed.
 */
class E5VariantAndConfigTest {

    // ── E5Variant.parse ────────────────────────────────────────────────

    @Test
    void parseDefaultsToBaseStsEnDe() {
        assertEquals(E5Variant.BASE_STS_EN_DE, E5Variant.parse(null));
        assertEquals(E5Variant.BASE_STS_EN_DE, E5Variant.parse(""));
        assertEquals(E5Variant.BASE_STS_EN_DE, E5Variant.parse("  "));
    }

    @Test
    void parseRecognisesCanonicalAndAliasTokens() {
        assertEquals(E5Variant.SMALL_V2, E5Variant.parse("small-v2"));
        assertEquals(E5Variant.SMALL_V2, E5Variant.parse("e5-small"));
        assertEquals(E5Variant.BASE_V2, E5Variant.parse("base-v2"));
        assertEquals(E5Variant.BASE_V2, E5Variant.parse("BASE"));
        assertEquals(E5Variant.LARGE_V2, E5Variant.parse("e5-large-v2"));
        assertEquals(E5Variant.BASE_STS_EN_DE, E5Variant.parse("base-sts-en-de"));
        assertEquals(E5Variant.BASE_STS_EN_DE, E5Variant.parse("en-de"));
    }

    @Test
    void parseRejectsUnknown() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> E5Variant.parse("jina"));
        assertTrue(ex.getMessage().contains("small-v2"));
        assertTrue(ex.getMessage().contains("base-sts-en-de"));
    }

    @Test
    void allVariantsValidateSuccessfully() {
        for (E5Variant v : E5Variant.values()) {
            v.config().validate();
            assertEquals(v.config().hiddenSize(), v.config().outputDimension(),
                    v + ": outputDimension must equal hiddenSize");
            assertEquals(PoolingStrategy.MEAN, v.config().poolingStrategy());
            assertEquals("gelu", v.config().hiddenAct());
        }
    }

    // ── BertConfigJson read + verifyMatches ────────────────────────────

    @Test
    void readsHuggingFaceConfigJson(@TempDir Path tmp) throws IOException, EmbeddingException {
        // BERT-base shape – matches E5 base/base-sts.
        writeConfig(tmp, """
                {
                  "_name_or_path": "intfloat/e5-base-v2",
                  "hidden_size": 768,
                  "num_hidden_layers": 12,
                  "num_attention_heads": 12,
                  "intermediate_size": 3072,
                  "max_position_embeddings": 512,
                  "type_vocab_size": 2,
                  "vocab_size": 30522,
                  "layer_norm_eps": 1e-12,
                  "hidden_act": "gelu"
                }
                """);
        BertEncoderConfig c = BertConfigJson.read(tmp, "fallback",
                PoolingStrategy.MEAN, true);
        assertEquals("intfloat/e5-base-v2", c.modelName());
        assertEquals(768, c.hiddenSize());
        assertEquals(12, c.numLayers());
        assertEquals(12, c.numHeads());
        assertEquals(64, c.headDim());
        assertEquals(3072, c.intermediateSize());
        assertEquals(30522, c.vocabSize());
        assertEquals(768, c.outputDimension());
    }

    @Test
    void normalisesGeluVariants(@TempDir Path tmp) throws IOException, EmbeddingException {
        writeConfig(tmp, """
                {
                  "hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 12,
                  "intermediate_size": 1536, "max_position_embeddings": 512,
                  "type_vocab_size": 2, "vocab_size": 30522, "layer_norm_eps": 1e-12,
                  "hidden_act": "gelu_new"
                }
                """);
        BertEncoderConfig c = BertConfigJson.read(tmp, "fb", PoolingStrategy.MEAN, true);
        assertEquals("gelu", c.hiddenAct());
    }

    @Test
    void missingConfigJsonIsAClearError(@TempDir Path tmp) {
        EmbeddingException ex = assertThrows(EmbeddingException.class,
                () -> BertConfigJson.read(tmp, "fb", PoolingStrategy.MEAN, true));
        assertTrue(ex.getMessage().contains("config.json"));
    }

    @Test
    void verifyMatchesRejectsShapeMismatch(@TempDir Path tmp) throws IOException, EmbeddingException {
        // On-disk = base shape; declared = small shape → must reject.
        writeConfig(tmp, """
                {
                  "_name_or_path": "intfloat/e5-base-v2",
                  "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                  "intermediate_size": 3072, "max_position_embeddings": 512,
                  "type_vocab_size": 2, "vocab_size": 30522, "layer_norm_eps": 1e-12,
                  "hidden_act": "gelu"
                }
                """);
        BertEncoderConfig onDisk = BertConfigJson.read(tmp, "fb", PoolingStrategy.MEAN, true);
        BertEncoderConfig declared = E5Variant.SMALL_V2.config(); // hidden=384

        EmbeddingException ex = assertThrows(EmbeddingException.class,
                () -> BertConfigJson.verifyMatches(declared, onDisk, tmp));
        assertTrue(ex.getMessage().contains("config mismatch"),
                "should call out the mismatch: " + ex.getMessage());
        assertTrue(ex.getMessage().contains("e5.model") || ex.getMessage().contains("e5.modelDir"),
                "should hint at the relevant system properties: " + ex.getMessage());
    }

    @Test
    void verifyMatchesAcceptsShapeAgreement(@TempDir Path tmp) throws IOException, EmbeddingException {
        writeConfig(tmp, """
                {
                  "_name_or_path": "intfloat/e5-base-v2",
                  "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                  "intermediate_size": 3072, "max_position_embeddings": 512,
                  "type_vocab_size": 2, "vocab_size": 30522, "layer_norm_eps": 1e-12,
                  "hidden_act": "gelu"
                }
                """);
        BertEncoderConfig onDisk = BertConfigJson.read(tmp, "fb", PoolingStrategy.MEAN, true);
        assertDoesNotThrow(() ->
                BertConfigJson.verifyMatches(E5Variant.BASE_V2.config(), onDisk, tmp));
    }

    // ── strict validate() invariants ───────────────────────────────────

    @Test
    void validateRejectsNonGeluActivation() {
        BertEncoderConfig bad = new BertEncoderConfig(
                "test", 24, 1, 4, 48, 16, 2, 32, 1e-12f,
                "relu", 24, PoolingStrategy.MEAN, true);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, bad::validate);
        assertTrue(ex.getMessage().toLowerCase().contains("gelu"));
    }

    @Test
    void validateRejectsNonMeanPooling() {
        BertEncoderConfig bad = new BertEncoderConfig(
                "test", 24, 1, 4, 48, 16, 2, 32, 1e-12f,
                "gelu", 24, PoolingStrategy.CLS, true);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, bad::validate);
        assertTrue(ex.getMessage().toLowerCase().contains("mean"));
    }

    @Test
    void validateRejectsProjectionHead() {
        // outputDimension != hiddenSize would require a dense projection.
        BertEncoderConfig bad = new BertEncoderConfig(
                "test", 24, 1, 4, 48, 16, 2, 32, 1e-12f,
                "gelu", 12 /* != 24 */, PoolingStrategy.MEAN, true);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, bad::validate);
        assertTrue(ex.getMessage().toLowerCase().contains("outputdimension"));
    }

    private static void writeConfig(Path tmp, String json) throws IOException {
        Files.writeString(tmp.resolve("config.json"), json);
    }
}

