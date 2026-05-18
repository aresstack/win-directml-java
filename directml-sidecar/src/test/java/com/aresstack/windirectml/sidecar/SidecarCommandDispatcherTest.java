package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcRequest;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcResponse;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class SidecarCommandDispatcherTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private JsonRpcRequest request(String method) throws Exception {
        String json = "{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"" + method + "\",\"params\":{}}";
        return mapper.readValue(json, JsonRpcRequest.class);
    }

    @Test
    void unknownMethodReturnsMethodNotFound() throws Exception {
        SidecarCommandDispatcher d = new SidecarCommandDispatcher();
        JsonRpcResponse resp = d.dispatch(request("doesNotExist"));
        assertNotNull(resp.error());
        assertEquals(JsonRpcErrorCode.METHOD_NOT_FOUND, resp.error().code());
        assertNull(resp.result());
    }

    @Test
    void typedMethodExceptionMapsToConfiguredCode() throws Exception {
        SidecarCommandDispatcher d = new SidecarCommandDispatcher();
        d.register("boom", params -> {
            throw new JsonRpcMethodException(JsonRpcErrorCode.MODEL_NOT_READY, "Not ready");
        });
        JsonRpcResponse resp = d.dispatch(request("boom"));
        assertNotNull(resp.error());
        assertEquals(JsonRpcErrorCode.MODEL_NOT_READY, resp.error().code());
        assertEquals("Not ready", resp.error().message());
    }

    @Test
    void uncheckedExceptionMapsToInternalError() throws Exception {
        SidecarCommandDispatcher d = new SidecarCommandDispatcher();
        d.register("crash", params -> {
            throw new RuntimeException("boom");
        });
        JsonRpcResponse resp = d.dispatch(request("crash"));
        assertNotNull(resp.error());
        assertEquals(JsonRpcErrorCode.INTERNAL_ERROR, resp.error().code());
    }

    @Test
    void successResponseCarriesHandlerResult() throws Exception {
        SidecarCommandDispatcher d = new SidecarCommandDispatcher();
        d.register("echo", params -> "pong");
        JsonRpcResponse resp = d.dispatch(request("echo"));
        assertNull(resp.error());
        assertEquals("pong", resp.result());
    }
}

