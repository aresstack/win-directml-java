package com.aresstack.windirectml.sidecar;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link EmbeddingModelRegistry} – verifies that the seven
 * company-list model IDs are present and classified as required by the
 * ticket (embedding vs decoder / summarizer, shipped vs planned, …).
 */
class EmbeddingModelRegistryTest {

    @Test
    void allSevenCompanyModelsArePresent() {
        List<String> expected = List.of(
                "sentence-transformers/all-MiniLM-L6-v2",
                "danielheinz/e5-base-sts-en-de",
                "openai/gpt-oss-120b",
                "jinaai/jina-embeddings-v2-base-de",
                "casperhansen/llama-3.3-70b-instruct-awq",
                "intfloat/multilingual-e5-large-instruct",
                "ellamind/summarizer-v6-llama-v2");
        for (String id : expected) {
            Optional<EmbeddingModelRegistry.Entry> e =
                    EmbeddingModelRegistry.findByModelId(id);
            assertTrue(e.isPresent(), "registry must contain " + id);
            assertEquals(id, e.get().modelId());
        }
    }

    @Test
    void lookupIsCaseInsensitiveAndTrimmed() {
        Optional<EmbeddingModelRegistry.Entry> a =
                EmbeddingModelRegistry.findByModelId("OPENAI/GPT-OSS-120B");
        Optional<EmbeddingModelRegistry.Entry> b =
                EmbeddingModelRegistry.findByModelId("  openai/gpt-oss-120b  ");
        assertTrue(a.isPresent());
        assertTrue(b.isPresent());
        assertSame(a.get(), b.get());
    }

    @Test
    void unknownAndBlankIdsResolveToEmpty() {
        assertTrue(EmbeddingModelRegistry.findByModelId(null).isEmpty());
        assertTrue(EmbeddingModelRegistry.findByModelId("").isEmpty());
        assertTrue(EmbeddingModelRegistry.findByModelId("   ").isEmpty());
        assertTrue(EmbeddingModelRegistry.findByModelId("minilm").isEmpty(),
                "short aliases are not full registry IDs");
        assertTrue(EmbeddingModelRegistry.findByModelId("some/unknown-model").isEmpty());
    }

    @Test
    void shippedEmbeddingsExposeAnImplementationFamily() {
        EmbeddingModelRegistry.Entry minilm = EmbeddingModelRegistry
                .findByModelId("sentence-transformers/all-MiniLM-L6-v2").orElseThrow();
        EmbeddingModelRegistry.Entry e5 = EmbeddingModelRegistry
                .findByModelId("danielheinz/e5-base-sts-en-de").orElseThrow();

        assertEquals(EmbeddingModelRegistry.UseCase.EMBEDDING, minilm.useCase());
        assertEquals(EmbeddingModelRegistry.Status.SHIPPED, minilm.status());
        assertEquals("minilm", minilm.embedFamily());

        assertEquals(EmbeddingModelRegistry.UseCase.EMBEDDING, e5.useCase());
        assertEquals(EmbeddingModelRegistry.Status.SHIPPED, e5.status());
        assertEquals("e5", e5.embedFamily());
    }

    @Test
    void plannedEmbeddingsHaveNoFamilyHook() {
        EmbeddingModelRegistry.Entry jina = EmbeddingModelRegistry
                .findByModelId("jinaai/jina-embeddings-v2-base-de").orElseThrow();
        EmbeddingModelRegistry.Entry mlE5 = EmbeddingModelRegistry
                .findByModelId("intfloat/multilingual-e5-large-instruct").orElseThrow();

        assertEquals(EmbeddingModelRegistry.UseCase.EMBEDDING, jina.useCase());
        assertEquals(EmbeddingModelRegistry.Status.PLANNED, jina.status());
        assertNull(jina.embedFamily(), "planned embeddings must not claim a family");

        assertEquals(EmbeddingModelRegistry.UseCase.EMBEDDING, mlE5.useCase());
        assertEquals(EmbeddingModelRegistry.Status.PLANNED, mlE5.status());
        assertNull(mlE5.embedFamily());
    }

    @Test
    void nonEmbeddingModelsAreClassifiedAsDecoderOrSummarizer() {
        EmbeddingModelRegistry.Entry gpt = EmbeddingModelRegistry
                .findByModelId("openai/gpt-oss-120b").orElseThrow();
        EmbeddingModelRegistry.Entry llama = EmbeddingModelRegistry
                .findByModelId("casperhansen/llama-3.3-70b-instruct-awq").orElseThrow();
        EmbeddingModelRegistry.Entry summ = EmbeddingModelRegistry
                .findByModelId("ellamind/summarizer-v6-llama-v2").orElseThrow();

        assertEquals(EmbeddingModelRegistry.UseCase.DECODER, gpt.useCase());
        assertFalse(gpt.isEmbedding());
        assertNull(gpt.embedFamily());

        assertEquals(EmbeddingModelRegistry.UseCase.DECODER, llama.useCase());
        assertFalse(llama.isEmbedding());

        assertEquals(EmbeddingModelRegistry.UseCase.SUMMARIZER, summ.useCase());
        assertFalse(summ.isEmbedding());
    }

    @Test
    void decoderErrorMessageMatchesTicketContract() {
        EmbeddingModelRegistry.Entry gpt = EmbeddingModelRegistry
                .findByModelId("openai/gpt-oss-120b").orElseThrow();
        String msg = EmbeddingModelRegistry.nonEmbeddingErrorMessage(gpt);
        // Exact wording from the ticket so downstream tooling can match on it.
        assertEquals(
                "Model openai/gpt-oss-120b is not an embedding model. "
                        + "Decoder models are not supported by the embed endpoint.",
                msg);
    }

    @Test
    void summarizerErrorMessageMentionsSummarizer() {
        EmbeddingModelRegistry.Entry summ = EmbeddingModelRegistry
                .findByModelId("ellamind/summarizer-v6-llama-v2").orElseThrow();
        String msg = EmbeddingModelRegistry.nonEmbeddingErrorMessage(summ);
        assertTrue(msg.contains("ellamind/summarizer-v6-llama-v2"));
        assertTrue(msg.contains("not an embedding model"));
        assertTrue(msg.contains("Summarizer"));
    }

    @Test
    void unimplementedEmbeddingMessageNamesStatus() {
        EmbeddingModelRegistry.Entry jina = EmbeddingModelRegistry
                .findByModelId("jinaai/jina-embeddings-v2-base-de").orElseThrow();
        String msg = EmbeddingModelRegistry.unimplementedEmbeddingErrorMessage(jina);
        assertTrue(msg.contains("jinaai/jina-embeddings-v2-base-de"));
        assertTrue(msg.contains("planned"));
        assertTrue(msg.contains("SUPPORTED_MODELS.md"));
    }

    @Test
    void entriesPreserveDeclarationOrder() {
        List<EmbeddingModelRegistry.Entry> all = EmbeddingModelRegistry.entries();
        assertEquals(7, all.size());
        assertEquals("sentence-transformers/all-MiniLM-L6-v2", all.get(0).modelId());
        assertEquals("ellamind/summarizer-v6-llama-v2", all.get(all.size() - 1).modelId());
    }

    @Test
    void shippedEntriesCarryRequiredMetadata() {
        EmbeddingModelRegistry.Entry minilm = EmbeddingModelRegistry
                .findByModelId("sentence-transformers/all-MiniLM-L6-v2").orElseThrow();
        assertNotNull(minilm.provider());
        assertNotNull(minilm.architecture());
        assertNotNull(minilm.tokenizerType());
        assertNotNull(minilm.backendSupport());
        assertFalse(minilm.modelDirHints().isEmpty(),
                "shipped embedding entries must declare modelDirHints");
        assertNotNull(minilm.downloadScriptSupport());
        assertNotNull(minilm.realModelTestStatus());
    }
}
