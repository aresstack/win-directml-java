package com.aresstack.windirectml.sidecar;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for {@link DirectMlPhi3Sidecar#embedFamily(String)} – the
 * stable parser for {@code -Dembed.model}.
 */
class EmbedFamilyParseTest {

    @Test
    void nullAndBlankDefaultToMinilm() {
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily(null));
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily(""));
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily("   "));
    }

    @Test
    void recognisedAliasesMapToCanonicalToken() {
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily("minilm"));
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily("MiniLM"));
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily("all-MiniLM-L6-v2"));
        assertEquals("e5", DirectMlPhi3Sidecar.embedFamily("e5"));
        assertEquals("e5", DirectMlPhi3Sidecar.embedFamily("E5"));
        assertEquals("e5", DirectMlPhi3Sidecar.embedFamily("e5-base-sts-en-de"));
    }

    @Test
    void unknownFamilyRejected() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> DirectMlPhi3Sidecar.embedFamily("jina"));
        org.junit.jupiter.api.Assertions.assertTrue(ex.getMessage().contains("e5"));
        org.junit.jupiter.api.Assertions.assertTrue(ex.getMessage().contains("minilm"));
    }

    @Test
    void fullEmbeddingModelIdsResolveViaRegistry() {
        assertEquals("minilm", DirectMlPhi3Sidecar.embedFamily(
                "sentence-transformers/all-MiniLM-L6-v2"));
    }

    @Test
    void danielheinzE5BaseStsEnDeRejectedAsUnimplemented() {
        // The upstream checkpoint at huggingface.co/danielheinz/e5-base-sts-en-de
        // hosts an XLMRobertaModel (vocab=250002, type_vocab_size=1) and does
        // not run on the current WordPiece-only E5 path. The registry classifies
        // it as planned with embedFamily=null so the gate rejects it with the
        // standard "planned" / SUPPORTED_MODELS.md message.
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> DirectMlPhi3Sidecar.embedFamily(
                        "danielheinz/e5-base-sts-en-de"));
        org.junit.jupiter.api.Assertions.assertTrue(
                ex.getMessage().contains("danielheinz/e5-base-sts-en-de"),
                ex.getMessage());
        org.junit.jupiter.api.Assertions.assertTrue(
                ex.getMessage().contains("planned"), ex.getMessage());
        org.junit.jupiter.api.Assertions.assertTrue(
                ex.getMessage().contains("SUPPORTED_MODELS.md"), ex.getMessage());
    }

    @Test
    void decoderModelIdRejectedWithExplicitMessage() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> DirectMlPhi3Sidecar.embedFamily("openai/gpt-oss-120b"));
        // Exact wording from the ticket – do not loosen without updating
        // both the ticket and any downstream tooling that matches on it.
        assertEquals(
                "Model openai/gpt-oss-120b is not an embedding model. "
                        + "Decoder models are not supported by the embed endpoint.",
                ex.getMessage());
    }

    @Test
    void summarizerModelIdRejected() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> DirectMlPhi3Sidecar.embedFamily("ellamind/summarizer-v6-llama-v2"));
        org.junit.jupiter.api.Assertions.assertTrue(
                ex.getMessage().contains("not an embedding model"),
                ex.getMessage());
        org.junit.jupiter.api.Assertions.assertTrue(
                ex.getMessage().contains("Summarizer"),
                ex.getMessage());
    }

    @Test
    void plannedEmbeddingModelIdRejectedAsUnimplemented() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> DirectMlPhi3Sidecar.embedFamily(
                        "intfloat/multilingual-e5-large-instruct"));
        org.junit.jupiter.api.Assertions.assertTrue(
                ex.getMessage().contains("planned"), ex.getMessage());
        org.junit.jupiter.api.Assertions.assertTrue(
                ex.getMessage().contains("SUPPORTED_MODELS.md"), ex.getMessage());
    }
}

