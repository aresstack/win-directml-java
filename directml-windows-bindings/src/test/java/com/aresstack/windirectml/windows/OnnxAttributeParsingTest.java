package com.aresstack.windirectml.windows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/** Verify standard ONNX AttributeProto wire fields required by Piper VITS. */
final class OnnxAttributeParsingTest {

    @TempDir
    Path directory;

    @Test
    void readsTensorScalarsAndRepeatedAttributesWithoutOnnxRuntime() throws Exception {
        byte[] floatTensor = message(
                varintField(1, 2),
                varintField(2, OnnxModelReader.ONNX_FLOAT),
                bytesField(9, ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                        .putFloat(1.25f).putFloat(-2.5f).array()));
        byte[] attributes = message(
                bytesField(5, attribute("value", bytesField(5, floatTensor))),
                bytesField(5, attribute("axis", varintField(3, -1))),
                bytesField(5, attribute("alpha", fixed32Field(2, 0.25f))),
                bytesField(5, attribute("mode", bytesField(4, "edge".getBytes(java.nio.charset.StandardCharsets.US_ASCII)))),
                bytesField(5, attribute("factors", bytesField(7, ByteBuffer.allocate(8)
                        .order(ByteOrder.LITTLE_ENDIAN).putFloat(1f).putFloat(0.5f).array()))),
                bytesField(5, attribute("pads", bytesField(8, message(varint(2), varint(3))))));
        byte[] node = message(bytesField(2, "out".getBytes(java.nio.charset.StandardCharsets.US_ASCII)),
                bytesField(4, "Constant".getBytes(java.nio.charset.StandardCharsets.US_ASCII)),
                attributes);
        byte[] graph = bytesField(1, node);
        Path model = directory.resolve("attributes.onnx");
        Files.write(model, bytesField(7, graph));

        OnnxModelReader.OnnxNode parsed = OnnxModelReader.parse(model).nodes().get(0);

        assertEquals("Constant", parsed.opType());
        assertEquals(-1, parsed.getInt("axis", 0));
        assertEquals(0.25f, (Float) parsed.attrs().get("alpha"), 0f);
        assertEquals("edge", parsed.getString("mode", ""));
        assertEquals(List.of(2L, 3L), parsed.getInts("pads"));
        assertEquals(List.of(1f, 0.5f), parsed.attrs().get("factors"));
        OnnxModelReader.OnnxTensor tensor = parsed.getTensor("value");
        assertNotNull(tensor);
        assertArrayEquals(new long[]{2}, tensor.dims());
        assertArrayEquals(new float[]{1.25f, -2.5f}, tensor.data(), 0f);
    }

    private static byte[] attribute(String name, byte[] content) throws IOException {
        return message(bytesField(1, name.getBytes(java.nio.charset.StandardCharsets.US_ASCII)), content);
    }

    private static byte[] fixed32Field(int number, float value) throws IOException {
        return message(varint(number << 3 | 5),
                ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
                        .putFloat(value).array());
    }

    private static byte[] bytesField(int number, byte[] bytes) throws IOException {
        return message(varint(number << 3 | 2), varint(bytes.length), bytes);
    }

    private static byte[] varintField(int number, long value) throws IOException {
        return message(varint(number << 3), varint(value));
    }

    private static byte[] varint(long value) {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        do {
            int next = (int) (value & 127);
            value >>>= 7;
            if (value != 0) {
                next |= 128;
            }
            output.write(next);
        } while (value != 0);
        return output.toByteArray();
    }

    private static byte[] message(byte[]... parts) throws IOException {
        ByteArrayOutputStream result = new ByteArrayOutputStream();
        for (byte[] part : parts) {
            result.write(part);
        }
        return result.toByteArray();
    }
}
