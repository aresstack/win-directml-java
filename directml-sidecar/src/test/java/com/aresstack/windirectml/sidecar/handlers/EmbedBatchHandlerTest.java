package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.config.InputLimits;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.sidecar.SidecarStatus;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbedBatchHandlerTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * Stub embedder that returns deterministic vectors derived from the
     * input text length. Records every request to assert prefix/normalize
     * propagation.
     */
    private static final class StubEmbedder implements EmbeddingModel {
        final List<EmbeddingRequest> seen = new ArrayList<>();
        boolean batchOverrideCalled = false;

        @Override
        public boolean isReady() {
            return true;
        }

        @Override
        public int dimension() {
            return 4;
        }

        @Override
        public EmbeddingVector embed(EmbeddingRequest request) {
            seen.add(request);
            float[] v = new float[]{request.text().length(), 0f, 0f, 0f};
            return new EmbeddingVector(v, 4, "stub/embed", request.normalize());
        }

        @Override
        public List<EmbeddingVector> embedBatch(List<EmbeddingRequest> requests) {
            batchOverrideCalled = true;
            List<EmbeddingVector> out = new ArrayList<>(requests.size());
            for (EmbeddingRequest r : requests) out.add(embed(r));
            return out;
        }
    }

    @Test
    void returnsNotImplementedWhenNoModelRegistered() {
        EmbedBatchHandler h = new EmbedBatchHandler(() -> null, new SidecarStatus());
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .set("texts", MAPPER.createArrayNode().add("hi"))));
        assertEquals(JsonRpcErrorCode.NOT_IMPLEMENTED, ex.code());
    }

    @Test
    void rejectsMissingTextsField() {
        EmbedBatchHandler h = new EmbedBatchHandler(StubEmbedder::new, new SidecarStatus());
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()));
        assertEquals(JsonRpcErrorCode.INVALID_PARAMS, ex.code());
    }

    @Test
    void rejectsEmptyTextsArray() {
        EmbedBatchHandler h = new EmbedBatchHandler(StubEmbedder::new, new SidecarStatus());
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .set("texts", MAPPER.createArrayNode())));
        assertEquals(JsonRpcErrorCode.INVALID_PARAMS, ex.code());
    }

    @Test
    void rejectsBlankTextEntry() {
        EmbedBatchHandler h = new EmbedBatchHandler(StubEmbedder::new, new SidecarStatus());
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .set("texts", MAPPER.createArrayNode().add("ok").add("  "))));
        assertEquals(JsonRpcErrorCode.INVALID_PARAMS, ex.code());
        assertTrue(ex.getMessage().contains("texts[1]"),
                "error must point at the offending index");
    }

    @Test
    @SuppressWarnings("unchecked")
    void happyPathReturnsVectorsInOrder() {
        StubEmbedder stub = new StubEmbedder();
        EmbedBatchHandler h = new EmbedBatchHandler(() -> stub, new SidecarStatus());

        JsonNode params = MAPPER.createObjectNode()
                .put("normalize", false)
                .put("prefix", "passage: ")
                .set("texts", MAPPER.createArrayNode().add("a").add("bcd").add("ef"));
        Object result = h.handle(params);
        Map<String, Object> m = (Map<String, Object>) result;

        assertEquals("stub/embed", m.get("model"));
        assertEquals(4, m.get("dimension"));
        assertEquals(3, m.get("count"));
        assertEquals(false, m.get("normalized"));

        List<float[]> vecs = (List<float[]>) m.get("vectors");
        assertEquals(3, vecs.size());
        // Stub encodes the bare text length into v[0]. prefix is kept on
        // the request but not concatenated into request.text().
        assertEquals(1f, vecs.get(0)[0], 1e-6f);
        assertEquals(3f, vecs.get(1)[0], 1e-6f);
        assertEquals(2f, vecs.get(2)[0], 1e-6f);

        // The handler must have forwarded the requests with prefix + normalize.
        assertTrue(stub.batchOverrideCalled, "handler must route through embedBatch()");
        assertEquals(3, stub.seen.size());
        for (EmbeddingRequest r : stub.seen) {
            assertEquals("passage: ", r.prefix());
            assertFalse(r.normalize());
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void defaultNormalizeAndAbsentPrefix() {
        StubEmbedder stub = new StubEmbedder();
        EmbedBatchHandler h = new EmbedBatchHandler(() -> stub, new SidecarStatus());

        Object result = h.handle(MAPPER.createObjectNode()
                .set("texts", MAPPER.createArrayNode().add("hello")));
        Map<String, Object> m = (Map<String, Object>) result;
        assertEquals(true, m.get("normalized"));
        assertEquals(1, m.get("count"));
        for (EmbeddingRequest r : stub.seen) {
            assertTrue(r.normalize());
            assertEquals(null, r.prefix());
        }
    }

    @Test
    void rejectsNonStringEntry() {
        EmbedBatchHandler h = new EmbedBatchHandler(StubEmbedder::new, new SidecarStatus());
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .set("texts", MAPPER.createArrayNode().add("ok").add(42))));
        assertEquals(JsonRpcErrorCode.INVALID_PARAMS, ex.code());
        assertTrue(ex.getMessage().contains("texts[1]"));
    }

    @Test
    void rejectsBatchSizeExceedingLimit() {
        EmbedBatchHandler h = new EmbedBatchHandler(StubEmbedder::new, new SidecarStatus());
        var textsNode = MAPPER.createArrayNode();
        for (int i = 0; i <= InputLimits.maxEmbedBatchSize(); i++) {
            textsNode.add("text " + i);
        }
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode().set("texts", textsNode)));
        assertEquals(JsonRpcErrorCode.LIMIT_EXCEEDED, ex.code());
        assertTrue(ex.getMessage().contains("exceeds maximum"));
    }

    @Test
    void rejectsTextExceedingMaxLength() {
        EmbedBatchHandler h = new EmbedBatchHandler(StubEmbedder::new, new SidecarStatus());
        String longText = "x".repeat(InputLimits.maxTextLength() + 1);
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .set("texts", MAPPER.createArrayNode().add(longText))));
        assertEquals(JsonRpcErrorCode.LIMIT_EXCEEDED, ex.code());
        assertTrue(ex.getMessage().contains("exceeds maximum"));
    }
}

