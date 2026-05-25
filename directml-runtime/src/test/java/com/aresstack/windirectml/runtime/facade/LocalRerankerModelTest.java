package com.aresstack.windirectml.runtime.facade;

import com.aresstack.windirectml.encoder.reranker.RerankException;
import com.aresstack.windirectml.encoder.reranker.RerankRequest;
import com.aresstack.windirectml.encoder.reranker.RerankResult;
import com.aresstack.windirectml.encoder.reranker.Reranker;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Facade-level unit tests for {@link LocalRerankerModel}.
 * Uses a stub reranker so no real model weights are needed.
 */
class LocalRerankerModelTest {

    /** Stub that scores documents by inverse string length. */
    private static final class StubReranker implements Reranker {
        private boolean ready = true;

        @Override public boolean isReady() { return ready; }
        @Override public String modelName() { return "stub-reranker"; }

        @Override
        public List<RerankResult> rerank(RerankRequest request) throws RerankException {
            List<RerankResult> results = new ArrayList<>();
            for (int i = 0; i < request.documents().size(); i++) {
                // Score inversely proportional to length
                double score = 100.0 / (request.documents().get(i).length() + 1);
                results.add(new RerankResult(i, score));
            }
            results.sort(Comparator.comparingDouble(RerankResult::score).reversed());
            int top = request.effectiveTopN();
            return top >= results.size() ? results : results.subList(0, top);
        }

        @Override public void close() { ready = false; }
    }

    @Test
    void rerankReturnsSortedResults() throws RerankException {
        var stub = new StubReranker();
        var model = new LocalRerankerModel(stub);

        List<RerankResult> results = model.rerank("query",
                List.of("short", "a very long document text"));
        assertEquals(2, results.size());
        // "short" (5 chars) should score higher than "a very long document text" (25 chars)
        assertTrue(results.get(0).score() > results.get(1).score());
        assertEquals(0, results.get(0).originalIndex()); // "short" is index 0
    }

    @Test
    void rerankWithTopNLimitsResults() throws RerankException {
        var stub = new StubReranker();
        var model = new LocalRerankerModel(stub);

        List<RerankResult> results = model.rerank("query",
                List.of("a", "bb", "ccc"), 2);
        assertEquals(2, results.size());
    }

    @Test
    void modelNameDelegates() {
        var stub = new StubReranker();
        var model = new LocalRerankerModel(stub);
        assertEquals("stub-reranker", model.modelName());
    }

    @Test
    void closeClosesUnderlying() {
        var stub = new StubReranker();
        var model = new LocalRerankerModel(stub);
        assertTrue(model.isReady());
        model.close();
        assertFalse(model.isReady());
    }

    @Test
    void rerankRejectsNullQuery() {
        var stub = new StubReranker();
        var model = new LocalRerankerModel(stub);
        assertThrows(NullPointerException.class,
                () -> model.rerank(null, List.of("doc")));
    }

    @Test
    void rerankRejectsNullDocuments() {
        var stub = new StubReranker();
        var model = new LocalRerankerModel(stub);
        assertThrows(NullPointerException.class,
                () -> model.rerank("query", null));
    }
}
