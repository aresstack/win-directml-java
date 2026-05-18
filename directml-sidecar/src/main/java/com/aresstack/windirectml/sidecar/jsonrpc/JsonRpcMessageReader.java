package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Line-based JSON-RPC 2.0 message reader.
 * <p>
 * Reads exactly one JSON object per line from the underlying {@link InputStream}.
 * Empty or whitespace-only lines are skipped silently. Lines that fail to parse
 * as JSON yield a {@link RawLine} with non-null {@link RawLine#parseError()},
 * so the dispatcher can answer with a standard {@code parse error} response
 * (the request id is unknown in that case).
 */
public final class JsonRpcMessageReader implements AutoCloseable {

    private final BufferedReader reader;
    private final ObjectMapper mapper;

    public JsonRpcMessageReader(InputStream in, ObjectMapper mapper) {
        this.reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        this.mapper = mapper;
    }

    /**
     * Read the next non-empty line as a JSON-RPC request.
     *
     * @return next message, or {@code null} on EOF.
     */
    public RawLine readNext() throws IOException {
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.isBlank()) continue;
            try {
                JsonNode root = mapper.readTree(line);
                JsonRpcRequest req = mapper.treeToValue(root, JsonRpcRequest.class);
                return new RawLine(line, req, null);
            } catch (Exception e) {
                return new RawLine(line, null, e);
            }
        }
        return null;
    }

    @Override
    public void close() throws IOException {
        reader.close();
    }

    /** Either a successfully parsed {@link JsonRpcRequest} or a parse error. */
    public record RawLine(String raw, JsonRpcRequest request, Throwable parseError) {
        public boolean hasError() { return parseError != null; }
    }
}

