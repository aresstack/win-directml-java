package com.aresstack.windirectml.config.models;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link EmbeddingModelRegistry} &ndash; verifies that the
 * seven company-list model IDs are present and classified as required
 * by the ticket (embedding vs decoder / summarizer, shipped vs
 * planned, &hellip;).
 *
 * <p>This test lives in {@code directml-config} so it also serves as
 * the Java-8-compatibility guard: it compiles with
 * {@code release = 8}, so any accidental use of post-Java-8 APIs in
 * the registry would break this build.
 */
class EmbeddingModelRegistryTest {

    @Test
    void allSevenCompanyModelsArePresent() {
        List<String> expected = Arrays.asList(
                "sentence-transformers/all-MiniLM-L6-v2",
                "danielheinz/e5-base-sts-en-de",
                "openai/gpt-oss-120b",
                "jinaai/jina-embeddings-v2-base-de",
                "casperhansen/llama-3.3-70b-instruct-awq",
                "intfloat/multilingual-e5-large-instruct",
                "ellamind/summarizer-v6-llama-v2");
        for (String id : expected) {
            EmbeddingModelRegistry.Entry e = EmbeddingModelRegistry.findByModelId(id);
            assertNotNull(e, "registry must contain " + id);
            assertEquals(id, e.modelId());
        }
    }

    @Test
    void lookupIsCaseInsensitiveAndTrimmed() {
        EmbeddingModelRegistry.Entry a =
                EmbeddingModelRegistry.findByModelId("OPENAI/GPT-OSS-120B");
        EmbeddingModelRegistry.Entry b =
                EmbeddingModelRegistry.findByModelId("  openai/gpt-oss-120b  ");
        assertNotNull(a);
        assertNotNull(b);
        assertSame(a, b);
    }

    @Test
    void unknownAndBlankIdsResolveToNull() {
        assertNull(EmbeddingModelRegistry.findByModelId(null));
        assertNull(EmbeddingModelRegistry.findByModelId(""));
        assertNull(EmbeddingModelRegistry.findByModelId("   "));
        assertNull(EmbeddingModelRegistry.findByModelId("minilm"),
                "short aliases are not full registry IDs");
        assertNull(EmbeddingModelRegistry.findByModelId("some/unknown-model"));
    }

    @Test
    void shippedEmbeddingsExposeAnImplementationFamily() {
        EmbeddingModelRegistry.Entry minilm = EmbeddingModelRegistry
                .findByModelId("sentence-transformers/all-MiniLM-L6-v2");
        EmbeddingModelRegistry.Entry e5 = EmbeddingModelRegistry
                .findByModelId("danielheinz/e5-base-sts-en-de");

        assertNotNull(minilm);
        assertNotNull(e5);

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
                .findByModelId("jinaai/jina-embeddings-v2-base-de");
        EmbeddingModelRegistry.Entry mlE5 = EmbeddingModelRegistry
                .findByModelId("intfloat/multilingual-e5-large-instruct");

        assertNotNull(jina);
        assertNotNull(mlE5);

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
                .findByModelId("openai/gpt-oss-120b");
        EmbeddingModelRegistry.Entry llama = EmbeddingModelRegistry
                .findByModelId("casperhansen/llama-3.3-70b-instruct-awq");
        EmbeddingModelRegistry.Entry summ = EmbeddingModelRegistry
                .findByModelId("ellamind/summarizer-v6-llama-v2");

        assertNotNull(gpt);
        assertNotNull(llama);
        assertNotNull(summ);

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
                .findByModelId("openai/gpt-oss-120b");
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
                .findByModelId("ellamind/summarizer-v6-llama-v2");
        String msg = EmbeddingModelRegistry.nonEmbeddingErrorMessage(summ);
        assertTrue(msg.contains("ellamind/summarizer-v6-llama-v2"));
        assertTrue(msg.contains("not an embedding model"));
        assertTrue(msg.contains("Summarizer"));
    }

    @Test
    void nonEmbeddingErrorMessageRejectsEmbeddingEntries() {
        EmbeddingModelRegistry.Entry minilm = EmbeddingModelRegistry
                .findByModelId("sentence-transformers/all-MiniLM-L6-v2");
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingModelRegistry.nonEmbeddingErrorMessage(minilm));
    }

    @Test
    void unimplementedEmbeddingMessageNamesStatus() {
        EmbeddingModelRegistry.Entry jina = EmbeddingModelRegistry
                .findByModelId("jinaai/jina-embeddings-v2-base-de");
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
                .findByModelId("sentence-transformers/all-MiniLM-L6-v2");
        EmbeddingModelRegistry.Entry e5 = EmbeddingModelRegistry
                .findByModelId("danielheinz/e5-base-sts-en-de");
        assertNotNull(minilm.provider());
        assertNotNull(e5);
        assertNotNull(e5.provider());
        assertNotNull(minilm.architecture());
        assertNotNull(e5.architecture());
        assertNotNull(minilm.tokenizerType());
        assertNotNull(e5.tokenizerType());
        assertNotNull(minilm.backendSupport());
        assertNotNull(e5.backendSupport());
        assertFalse(minilm.modelDirHints().isEmpty(),
                "shipped embedding entries must declare modelDirHints");
        assertFalse(e5.modelDirHints().isEmpty(),
                "shipped embedding entries must declare modelDirHints");
        assertNotNull(minilm.downloadScriptSupport());
        assertNotNull(e5.downloadScriptSupport());
        assertNotNull(minilm.realModelTestStatus());
        assertNotNull(e5.realModelTestStatus());
        assertEquals("cpu, directml", minilm.backendSupport());

        assertEquals("cpu, directml", e5.backendSupport());
        assertTrue(e5.notes().contains("query: "));
        assertTrue(e5.notes().contains("passage: "));
        String e5RealModelStatus = e5.realModelTestStatus().toLowerCase(java.util.Locale.ROOT);
        assertTrue(e5RealModelStatus.contains("real-model"));
    }

    @Test
    void entriesByUseCaseEmbeddingFiltersOutDecodersAndSummarizers() {
        List<EmbeddingModelRegistry.Entry> embeddings =
                EmbeddingModelRegistry.entriesByUseCase(EmbeddingModelRegistry.UseCase.EMBEDDING);
        // 4 embedding models: MiniLM, e5-base-sts-en-de, Jina, multilingual-E5.
        assertEquals(4, embeddings.size(),
                "entriesByUseCase(EMBEDDING) must drop decoder/summarizer entries");
        for (EmbeddingModelRegistry.Entry e : embeddings) {
            assertEquals(EmbeddingModelRegistry.UseCase.EMBEDDING, e.useCase(),
                    "non-embedding model leaked into the embedding-only view: "
                            + e.modelId());
        }
        // Spot-check that the known decoder/summarizer IDs are NOT in the list.
        for (EmbeddingModelRegistry.Entry e : embeddings) {
            assertFalse("openai/gpt-oss-120b".equals(e.modelId()));
            assertFalse("casperhansen/llama-3.3-70b-instruct-awq".equals(e.modelId()));
            assertFalse("ellamind/summarizer-v6-llama-v2".equals(e.modelId()));
        }
    }

    @Test
    void entriesByUseCaseDecoderReturnsBothDecoderIds() {
        List<EmbeddingModelRegistry.Entry> decoders =
                EmbeddingModelRegistry.entriesByUseCase(EmbeddingModelRegistry.UseCase.DECODER);
        assertEquals(2, decoders.size());
    }

    @Test
    void entriesByUseCaseNullRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingModelRegistry.entriesByUseCase(null));
    }

    @Test
    void multilingualE5InstructAnalysisIsPinned() {
        // Pins the documented analysis for
        // intfloat/multilingual-e5-large-instruct so the registry
        // entry can never silently drift away from
        // SUPPORTED_MODELS.md section 1.1.1 (#39-C).
        EmbeddingModelRegistry.Entry e = EmbeddingModelRegistry
                .findByModelId("intfloat/multilingual-e5-large-instruct");
        assertNotNull(e);

        // Classification: visible as an embedding entry but explicitly
        // planned (no real-model test, no shipped runtime).
        assertEquals(EmbeddingModelRegistry.UseCase.EMBEDDING, e.useCase());
        assertEquals(EmbeddingModelRegistry.Status.PLANNED, e.status());
        assertNull(e.embedFamily(),
                "planned multilingual-E5-instruct must not claim an embedFamily "
                        + "until SentencePiece + XLM-R support lands");

        // Tokenizer analysis: SentencePiece (XLM-R), not WordPiece.
        String tokenizer = e.tokenizerType();
        assertNotNull(tokenizer);
        assertTrue(tokenizer.contains("SentencePiece"),
                "tokenizerType must record SentencePiece: " + tokenizer);
        assertTrue(tokenizer.contains("XLM-R"),
                "tokenizerType must call out the XLM-R variant: " + tokenizer);

        // Architecture analysis: XLM-RoBERTa-large.
        String architecture = e.architecture();
        assertNotNull(architecture);
        assertTrue(architecture.contains("XLM-RoBERTa"),
                "architecture must identify the XLM-RoBERTa core: " + architecture);

        // Notes must call out the non-compatibility, the instruction
        // prefix and forward callers to SUPPORTED_MODELS.md.
        String notes = e.notes();
        assertNotNull(notes);
        assertTrue(notes.contains("NOT compatible"),
                "notes must state non-compatibility with the BERT/E5 core: " + notes);
        assertTrue(notes.contains("SentencePiece"), notes);
        assertTrue(notes.contains("Instruct:"),
                "notes must record the instruction-prefix format: " + notes);
        assertTrue(notes.contains("SUPPORTED_MODELS.md"),
                "notes must point callers at the analysis section: " + notes);

        // Real-model test status must explicitly say no real-model
        // test exists yet (no fake-support claim).
        String realModel = e.realModelTestStatus();
        assertNotNull(realModel);
        assertTrue(realModel.toLowerCase(java.util.Locale.ROOT).contains("not yet tested"),
                "realModelTestStatus must record the missing real-model test: "
                        + realModel);
    }
}
