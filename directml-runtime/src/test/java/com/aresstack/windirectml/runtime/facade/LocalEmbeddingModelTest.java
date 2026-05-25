package com.aresstack.windirectml.runtime.facade;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Facade-level unit tests for {@link LocalEmbeddingModel}.
 * Uses a stub encoder so no real model weights are needed.
 */
class LocalEmbeddingModelTest {

    /** Stub that returns a deterministic vector based on text length. */
    private static final class StubEmbeddingModel implements EmbeddingModel, AutoCloseable {
        private boolean ready = true;

        @Override public boolean isReady() { return ready; }
        @Override public int dimension() { return 3; }

        @Override
        public EmbeddingVector embed(EmbeddingRequest request) throws EmbeddingException {
            String text = request.prefix() != null ? request.prefix() + request.text() : request.text();
            float len = text.length();
            return new EmbeddingVector(new float[]{len, len * 2, len * 3}, 3, "stub", true);
        }

        @Override public void close() { ready = false; }
    }

    @Test
    void embedReturnsSingleVector() throws EmbeddingException {
        var stub = new StubEmbeddingModel();
        var model = new LocalEmbeddingModel(stub, null);

        float[] result = model.embed("hi");
        assertEquals(3, result.length);
        assertEquals(2f, result[0]); // "hi".length() == 2
    }

    @Test
    void embedWithPrefixPrependsToText() throws EmbeddingException {
        var stub = new StubEmbeddingModel();
        var model = new LocalEmbeddingModel(stub, "query: ");

        float[] result = model.embed("hi");
        // "query: " + "hi" = 9 chars
        assertEquals(9f, result[0]);
    }

    @Test
    void embedBatchPreservesOrder() throws EmbeddingException {
        var stub = new StubEmbeddingModel();
        var model = new LocalEmbeddingModel(stub, null);

        List<float[]> results = model.embedBatch(List.of("a", "bb", "ccc"));
        assertEquals(3, results.size());
        assertEquals(1f, results.get(0)[0]); // "a".length() == 1
        assertEquals(2f, results.get(1)[0]); // "bb".length() == 2
        assertEquals(3f, results.get(2)[0]); // "ccc".length() == 3
    }

    @Test
    void embedBatchRejectsEmptyList() {
        var stub = new StubEmbeddingModel();
        var model = new LocalEmbeddingModel(stub, null);
        assertThrows(IllegalArgumentException.class, () -> model.embedBatch(List.of()));
    }

    @Test
    void embedRejectsNull() {
        var stub = new StubEmbeddingModel();
        var model = new LocalEmbeddingModel(stub, null);
        assertThrows(NullPointerException.class, () -> model.embed(null));
    }

    @Test
    void dimensionDelegatesToUnderlying() {
        var stub = new StubEmbeddingModel();
        var model = new LocalEmbeddingModel(stub, null);
        assertEquals(3, model.dimension());
    }

    @Test
    void closeClosesUnderlying() {
        var stub = new StubEmbeddingModel();
        var model = new LocalEmbeddingModel(stub, null);
        assertTrue(model.isReady());
        model.close();
        assertFalse(model.isReady());
    }
}
