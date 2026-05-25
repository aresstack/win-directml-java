package com.aresstack.windirectml.sidecar.handlers;

import com.aresstack.windirectml.config.InputLimits;
import com.aresstack.windirectml.encoder.reranker.RerankRequest;
import com.aresstack.windirectml.encoder.reranker.RerankResult;
import com.aresstack.windirectml.encoder.reranker.Reranker;
import com.aresstack.windirectml.sidecar.SidecarStatus;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RerankHandlerTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Stub reranker that just echoes a constant ranking. */
    private static final class StubReranker implements Reranker {
        private final List<RerankRequest> seen = new ArrayList<>();
        @Override public boolean isReady() { return true; }
        @Override public String modelName() { return "stub/reranker"; }
        @Override
        public List<RerankResult> rerank(RerankRequest request) {
            seen.add(request);
            List<RerankResult> out = new ArrayList<>();
            // Inverse-position score so the *last* document wins – a real
            // test would compute something, but we just need a stable order.
            for (int i = 0; i < request.documents().size(); i++) {
                out.add(new RerankResult(i, request.documents().size() - i));
            }
            int top = request.effectiveTopN();
            return top >= out.size() ? out : out.subList(0, top);
        }
        @Override public void close() {}
    }

    @Test
    void returnsNotImplementedWhenNoRerankerRegistered() {
        RerankHandler h = new RerankHandler(() -> null, new SidecarStatus());
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .put("query", "q")
                        .set("documents", MAPPER.createArrayNode().add("d"))));
        assertEquals(JsonRpcErrorCode.NOT_IMPLEMENTED, ex.code());
    }

    @Test
    void rejectsMissingParams() {
        RerankHandler h = new RerankHandler(StubReranker::new, new SidecarStatus());
        // Missing both query and documents.
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()));
        assertEquals(JsonRpcErrorCode.INVALID_PARAMS, ex.code());
    }

    @Test
    void rejectsBlankQuery() {
        RerankHandler h = new RerankHandler(StubReranker::new, new SidecarStatus());
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .put("query", "   ")
                        .set("documents", MAPPER.createArrayNode().add("d"))));
        assertEquals(JsonRpcErrorCode.INVALID_PARAMS, ex.code());
    }

    @Test
    void rejectsEmptyDocumentsArray() {
        RerankHandler h = new RerankHandler(StubReranker::new, new SidecarStatus());
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .put("query", "q")
                        .set("documents", MAPPER.createArrayNode())));
        assertEquals(JsonRpcErrorCode.INVALID_PARAMS, ex.code());
    }

    @Test
    void rejectsBlankDocumentEntry() {
        RerankHandler h = new RerankHandler(StubReranker::new, new SidecarStatus());
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .put("query", "q")
                        .set("documents", MAPPER.createArrayNode().add("ok").add("   "))));
        assertEquals(JsonRpcErrorCode.INVALID_PARAMS, ex.code());
        assertTrue(ex.getMessage().contains("documents[1]"),
                "error message should point at the offending index");
    }

    @Test
    @SuppressWarnings("unchecked")
    void happyPathReturnsRankedItems() {
        StubReranker stub = new StubReranker();
        RerankHandler h = new RerankHandler(() -> stub, new SidecarStatus());

        JsonNode params = MAPPER.createObjectNode()
                .put("query", "hello")
                .put("topN", 2)
                .set("documents", MAPPER.createArrayNode().add("a").add("b").add("c"));
        Object result = h.handle(params);
        Map<String, Object> m = (Map<String, Object>) result;
        assertEquals("stub/reranker", m.get("model"));
        List<Map<String, Object>> items = (List<Map<String, Object>>) m.get("results");
        assertEquals(2, items.size(), "topN must trim to 2 entries");
        for (Map<String, Object> it : items) {
            assertNotNull(it.get("index"));
            assertNotNull(it.get("score"));
        }
        assertTrue(items.get(0).get("score") instanceof Number);
        // The handler must have forwarded the parsed documents in order.
        assertEquals(Arrays.asList("a", "b", "c"), stub.seen.get(0).documents());
        assertEquals(2, stub.seen.get(0).topN());
    }

    @Test
    void rejectsDocumentsExceedingLimit() {
        StubReranker stub = new StubReranker();
        RerankHandler h = new RerankHandler(() -> stub, new SidecarStatus());
        var docsNode = MAPPER.createArrayNode();
        for (int i = 0; i <= InputLimits.maxRerankDocuments(); i++) {
            docsNode.add("doc " + i);
        }
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .put("query", "test")
                        .set("documents", docsNode)));
        assertEquals(JsonRpcErrorCode.LIMIT_EXCEEDED, ex.code());
        assertTrue(ex.getMessage().contains("exceeds maximum"));
    }

    @Test
    void rejectsQueryExceedingMaxLength() {
        StubReranker stub = new StubReranker();
        RerankHandler h = new RerankHandler(() -> stub, new SidecarStatus());
        String longQuery = "x".repeat(InputLimits.maxTextLength() + 1);
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .put("query", longQuery)
                        .set("documents", MAPPER.createArrayNode().add("doc"))));
        assertEquals(JsonRpcErrorCode.LIMIT_EXCEEDED, ex.code());
        assertTrue(ex.getMessage().contains("query length"));
    }

    @Test
    void rejectsDocumentExceedingMaxLength() {
        StubReranker stub = new StubReranker();
        RerankHandler h = new RerankHandler(() -> stub, new SidecarStatus());
        String longDoc = "y".repeat(InputLimits.maxRerankDocumentLength() + 1);
        JsonRpcMethodException ex = assertThrows(JsonRpcMethodException.class,
                () -> h.handle(MAPPER.createObjectNode()
                        .put("query", "test")
                        .set("documents", MAPPER.createArrayNode().add(longDoc))));
        assertEquals(JsonRpcErrorCode.LIMIT_EXCEEDED, ex.code());
        assertTrue(ex.getMessage().contains("documents[0]"));
    }
}

