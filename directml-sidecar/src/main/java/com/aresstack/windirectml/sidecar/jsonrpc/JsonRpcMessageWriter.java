package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/**
 * Line-based JSON-RPC 2.0 message writer.
 * <p>
 * Every {@link JsonRpcResponse} or {@link JsonRpcNotification} is serialized
 * to exactly one line on the underlying {@link OutputStream}, followed by a
 * newline and an explicit flush. Thread-safe; concurrent writers are
 * serialized by the writer's internal monitor.
 * <p>
 * <b>Important:</b> stdout must only ever carry JSON-RPC messages. Logs and
 * diagnostics belong on stderr.
 */
public final class JsonRpcMessageWriter implements AutoCloseable {

    private final BufferedWriter writer;
    private final ObjectMapper mapper;
    private final Object lock = new Object();

    public JsonRpcMessageWriter(OutputStream out, ObjectMapper mapper) {
        this.writer = new BufferedWriter(new OutputStreamWriter(out, StandardCharsets.UTF_8));
        this.mapper = mapper;
    }

    public void writeResponse(JsonRpcResponse response) {
        writeJson(response);
    }

    public void writeNotification(JsonRpcNotification notification) {
        writeJson(notification);
    }

    private void writeJson(Object message) {
        try {
            String line = mapper.writeValueAsString(message);
            synchronized (lock) {
                writer.write(line);
                writer.newLine();
                writer.flush();
            }
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to serialize JSON-RPC message", e);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void close() throws IOException {
        synchronized (lock) {
            writer.flush();
            writer.close();
        }
    }
}

