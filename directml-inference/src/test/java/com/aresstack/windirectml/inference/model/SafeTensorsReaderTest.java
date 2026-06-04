package com.aresstack.windirectml.inference.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class SafeTensorsReaderTest {

    @TempDir
    Path tempDir;

    @Test
    void parsesHeaderAndMmapBackedPayload() throws Exception {
        Path file = tempDir.resolve("model.safetensors");
        String header = "{\"model.embed_tokens.weight\":{" +
                "\"dtype\":\"F16\",\"shape\":[2,2],\"data_offsets\":[0,8]" +
                "},\"__metadata__\":{\"format\":\"pt\"}}";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer out = ByteBuffer.allocate(Long.BYTES + headerBytes.length + 8).order(ByteOrder.LITTLE_ENDIAN);
        out.putLong(headerBytes.length);
        out.put(headerBytes);
        out.put(new byte[]{1, 2, 3, 4, 5, 6, 7, 8});
        Files.write(file, out.array());

        SafeTensorsReader.SafeTensorsFile parsed = SafeTensorsReader.read(file);

        assertEquals("pt", parsed.metadata().get("format"));
        SafeTensorsReader.SafeTensorEntry tensor = parsed.tensors().get("model.embed_tokens.weight");
        assertNotNull(tensor);
        assertEquals("F16", tensor.dtype());
        assertEquals(10, tensor.onnxDataType());
        assertArrayEquals(new long[]{2, 2}, tensor.shape());
        assertEquals(8, tensor.byteLength());
        ByteBuffer data = tensor.dataBuffer();
        assertEquals(1, data.get(0));
        assertEquals(8, data.get(7));
    }

    @Test
    void rejectsInvalidByteLength() throws Exception {
        Path file = tempDir.resolve("bad.safetensors");
        String header = "{\"x\":{" +
                "\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]" +
                "}}";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer out = ByteBuffer.allocate(Long.BYTES + headerBytes.length + 4).order(ByteOrder.LITTLE_ENDIAN);
        out.putLong(headerBytes.length);
        out.put(headerBytes);
        out.putInt(1234);
        Files.write(file, out.array());

        java.io.IOException thrown = assertThrows(java.io.IOException.class, () -> SafeTensorsReader.read(file));
        assertTrue(thrown.getMessage().contains("byte length mismatch"));
    }
}

