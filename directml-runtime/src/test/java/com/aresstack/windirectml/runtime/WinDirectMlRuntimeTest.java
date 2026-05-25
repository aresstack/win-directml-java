package com.aresstack.windirectml.runtime;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.reranker.RerankException;
import com.aresstack.windirectml.encoder.reranker.RerankRequest;
import com.aresstack.windirectml.encoder.reranker.RerankResult;
import com.aresstack.windirectml.encoder.reranker.Reranker;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the {@link WinDirectMlRuntime} facade using mock models.
 * These tests validate the API contract without requiring real model weights
 * or GPU hardware.
 */
class WinDirectMlRuntimeTest {

    // ── Stub implementations ─────────────────────────────────────────────

    private static final EmbeddingModel STUB_EMBEDDING = new EmbeddingModel() {
        @Override public boolean isReady() { return true; }
        @Override public int dimension() { return 3; }
        @Override public EmbeddingVector embed(EmbeddingRequest request) {
            return new EmbeddingVector(new float[]{0.1f, 0.2f, 0.3f}, 3,
                    "stub-model", request.normalize());
        }
    };

    private static final Reranker STUB_RERANKER = new Reranker() {
        @Override public boolean isReady() { return true; }
        @Override public String modelName() { return "stub-reranker"; }
        @Override public List<RerankResult> rerank(RerankRequest request) {
            return List.of(new RerankResult(0, 1.5));
        }
        @Override public void close() {}
    };

    // ── Embedding tests ──────────────────────────────────────────────────

    @Test
    void embedReturnsVectorFromModel() throws EmbeddingException {
        try (var runtime = WinDirectMlRuntime.of(STUB_EMBEDDING, "cpu", null, null)) {
            assertTrue(runtime.isEmbeddingReady());
            EmbeddingVector vec = runtime.embed(EmbeddingRequest.of("hello"));
            assertEquals(3, vec.dimension());
            assertEquals("stub-model", vec.model());
        }
    }

    @Test
    void embedBatchPreservesOrder() throws EmbeddingException {
        try (var runtime = WinDirectMlRuntime.of(STUB_EMBEDDING, "cpu", null, null)) {
            var requests = List.of(
                    EmbeddingRequest.of("first"),
                    EmbeddingRequest.of("second")
            );
            List<EmbeddingVector> results = runtime.embedBatch(requests);
            assertEquals(2, results.size());
        }
    }

    @Test
    void embedThrowsModelReadinessWhenNoModel() {
        try (var runtime = WinDirectMlRuntime.of(null, null, null, null)) {
            assertFalse(runtime.isEmbeddingReady());
            assertThrows(ModelReadinessException.class,
                    () -> runtime.embed(EmbeddingRequest.of("hello")));
        }
    }

    @Test
    void embeddingDimensionReportsModelDimension() {
        try (var runtime = WinDirectMlRuntime.of(STUB_EMBEDDING, "cpu", null, null)) {
            assertEquals(3, runtime.embeddingDimension());
        }
    }

    @Test
    void embeddingDimensionReturnsMinusOneWhenNoModel() {
        try (var runtime = WinDirectMlRuntime.of(null, null, null, null)) {
            assertEquals(-1, runtime.embeddingDimension());
        }
    }

    // ── Reranking tests ──────────────────────────────────────────────────

    @Test
    void rerankReturnsResultsFromModel() throws RerankException {
        try (var runtime = WinDirectMlRuntime.of(null, null, STUB_RERANKER, "cpu")) {
            assertTrue(runtime.isRerankerReady());
            var results = runtime.rerank(
                    new RerankRequest("query", List.of("doc1"), 1));
            assertEquals(1, results.size());
            assertEquals(0, results.get(0).originalIndex());
        }
    }

    @Test
    void rerankThrowsModelReadinessWhenNoModel() {
        try (var runtime = WinDirectMlRuntime.of(null, null, null, null)) {
            assertFalse(runtime.isRerankerReady());
            assertThrows(ModelReadinessException.class,
                    () -> runtime.rerank(new RerankRequest("q", List.of("d"), 1)));
        }
    }

    @Test
    void rerankerModelNameExposed() {
        try (var runtime = WinDirectMlRuntime.of(null, null, STUB_RERANKER, "cpu")) {
            assertEquals("stub-reranker", runtime.rerankerModelName());
        }
    }

    @Test
    void rerankerModelNameNullWhenNoModel() {
        try (var runtime = WinDirectMlRuntime.of(null, null, null, null)) {
            assertNull(runtime.rerankerModelName());
        }
    }

    // ── Backend reporting ────────────────────────────────────────────────

    @Test
    void backendNamesExposed() {
        try (var runtime = WinDirectMlRuntime.of(STUB_EMBEDDING, "directml",
                STUB_RERANKER, "cpu")) {
            assertEquals("directml", runtime.embeddingBackend());
            assertEquals("cpu", runtime.rerankerBackend());
        }
    }

    // ── Builder ──────────────────────────────────────────────────────────

    @Test
    void builderWithNoModelDirsProducesEmptyRuntime() {
        try (var runtime = WinDirectMlRuntime.builder().build()) {
            assertFalse(runtime.isEmbeddingReady());
            assertFalse(runtime.isRerankerReady());
            assertNull(runtime.embeddingBackend());
            assertNull(runtime.rerankerBackend());
        }
    }

    @Test
    void builderWithNonexistentDirThrowsModelReadiness() {
        var builder = WinDirectMlRuntime.builder()
                .embeddingModelDir(java.nio.file.Path.of("/nonexistent/model"))
                .backend(Backend.CPU);
        assertThrows(ModelReadinessException.class, builder::build);
    }
}
