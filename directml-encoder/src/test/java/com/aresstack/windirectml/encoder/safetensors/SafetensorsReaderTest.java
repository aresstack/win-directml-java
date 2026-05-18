package com.aresstack.windirectml.encoder.safetensors;

import com.aresstack.windirectml.runtime.TensorDataType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SafetensorsReaderTest {

    @Test
    void readsSyntheticFile(@TempDir Path dir) throws Exception {
        float[] values = {1f, 2f, 3f, 4f, 5f, 6f};
        Path file = writeSafetensors(dir.resolve("syn.safetensors"), values);

        try (SafetensorsReader reader = SafetensorsReader.open(file)) {
            assertTrue(reader.tensorNames().contains("weights"));
            SafetensorsEntry e = reader.entry("weights");
            assertEquals(TensorDataType.FLOAT32, e.dataType());
            assertEquals(2, e.shape().rank());
            assertEquals(2, e.shape().dim(0));
            assertEquals(3, e.shape().dim(1));
            assertArrayEquals(values, reader.readFloat32("weights"), 0f);
        }
    }

    @Test
    void rejectsInvalidHeaderLength(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("bad.safetensors");
        ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(-1L);
        Files.write(file, buf.array());
        assertThrows(SafetensorsException.class, () -> SafetensorsReader.open(file));
    }

    @Test
    void rejectsUnknownDtype(@TempDir Path dir) throws Exception {
        String json = "{\"weights\":{\"dtype\":\"BANANA\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        byte[] jsonBytes = json.getBytes(StandardCharsets.UTF_8);
        Path file = dir.resolve("bad-dtype.safetensors");
        try (var out = Files.newOutputStream(file)) {
            ByteBuffer header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            header.putLong(jsonBytes.length);
            out.write(header.array());
            out.write(jsonBytes);
            out.write(new byte[8]);
        }
        assertThrows(SafetensorsException.class, () -> SafetensorsReader.open(file));
    }

    private static Path writeSafetensors(Path file, float[] floats) throws IOException {
        int dataLen = floats.length * 4;
        String json = "{\"weights\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0," + dataLen + "]}}";
        byte[] jsonBytes = json.getBytes(StandardCharsets.UTF_8);

        ByteBuffer headerLen = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        headerLen.putLong(jsonBytes.length);

        ByteBuffer dataBuf = ByteBuffer.allocate(dataLen).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : floats) dataBuf.putFloat(f);

        try (var out = Files.newOutputStream(file)) {
            out.write(headerLen.array());
            out.write(jsonBytes);
            out.write(dataBuf.array());
        }
        return file;
    }
}

