package com.aresstack.windirectml.sidecar.jsonrpc;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JsonRpcMessageReaderWriterTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void readerSkipsBlankLinesAndStopsAtEof() throws Exception {
        String input = "\n   \n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"a\"}\n\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"b\"}\n";
        try (JsonRpcMessageReader reader = new JsonRpcMessageReader(
                new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8)), mapper)) {
            JsonRpcMessageReader.RawLine first = reader.readNext();
            assertNotNull(first);
            assertEquals("a", first.request().method());

            JsonRpcMessageReader.RawLine second = reader.readNext();
            assertNotNull(second);
            assertEquals("b", second.request().method());

            assertNull(reader.readNext());
        }
    }

    @Test
    void readerReportsParseErrorWithoutThrowing() throws Exception {
        String input = "{not-json\n";
        try (JsonRpcMessageReader reader = new JsonRpcMessageReader(
                new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8)), mapper)) {
            JsonRpcMessageReader.RawLine line = reader.readNext();
            assertNotNull(line);
            assertTrue(line.hasError());
        }
    }

    @Test
    void writerProducesExactlyOneLinePerMessage() throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (JsonRpcMessageWriter writer = new JsonRpcMessageWriter(out, mapper)) {
            writer.writeResponse(JsonRpcResponse.success(mapper.valueToTree("1"), "ok"));
            writer.writeNotification(JsonRpcNotification.of("sidecar.started", null));
        }
        String text = out.toString(StandardCharsets.UTF_8);
        String[] lines = text.split("\\R");
        assertEquals(2, lines.length, "expected exactly two lines, got: " + text);
        for (String line : lines) {
            assertTrue(line.startsWith("{") && line.endsWith("}"),
                    "each line must be a single JSON object: " + line);
        }
    }
}

