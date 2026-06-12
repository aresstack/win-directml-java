package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.RuntimeTensor;
import com.aresstack.windirectml.windows.OnnxModelReader;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Device-free tests for the heap-light FP32 source seam on {@link T5TensorData} (slice H2b):
 * a FLOAT32 runtime tensor defers its float decode and exposes the little-endian mmap slice for direct
 * WARP upload, while FLOAT16 / reference tensors keep eager {@code float[]} values and no FP32 source.
 */
class T5TensorDataTest {

    private static RuntimeTensor fp32Tensor(String name, long[] dims, float[] values) {
        ByteBuffer raw = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) {
            raw.putFloat(v);
        }
        raw.flip();
        return new RuntimeTensor(name, dims, OnnxModelReader.ONNX_FLOAT, raw, values.length * Float.BYTES);
    }

    @Test
    void fp32TensorExposesLittleEndianSourceAndDecodesToSameValues() {
        float[] values = {1.0f, -2.0f, 3.5f, 0.25f, -0.5f, 4.0f};
        T5TensorData data = T5TensorData.from(fp32Tensor("w", new long[]{2, 3}, values));

        ByteBuffer source = data.fp32LittleEndianSource();
        assertNotNull(source, "FP32 tensor must expose a direct upload source");
        assertEquals(ByteOrder.LITTLE_ENDIAN, source.order());
        assertEquals(values.length * Float.BYTES, source.remaining());

        // Lazy float decode must reproduce the original values exactly.
        assertArrayEquals(values, data.values(), 0.0f);
        assertEquals(3.5f, data.at(2), 0.0f);
        assertEquals(4.0f, data.at(1, 2), 0.0f); // row 1, col 2
        assertEquals(6, data.elementCount());
    }

    @Test
    void fp32SourceIsIndependentViewPerCall() {
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        T5TensorData data = T5TensorData.from(fp32Tensor("w", new long[]{2, 2}, values));
        ByteBuffer a = data.fp32LittleEndianSource();
        a.position(8); // mutate one view
        ByteBuffer b = data.fp32LittleEndianSource();
        assertEquals(0, b.position(), "each fp32LittleEndianSource() call must return an independent view");
        assertEquals(values.length * Float.BYTES, b.remaining());
    }

    @Test
    void referenceTensorHasNoFp32SourceButKeepsValues() {
        float[] values = {0.0f, 1.0f, 2.0f, 3.0f};
        T5TensorData data = T5TensorData.reference("ref", new long[]{2, 2}, values);
        assertNull(data.fp32LittleEndianSource(), "reference/fused tensors must use the float[] fallback");
        assertArrayEquals(values, data.values(), 0.0f);
    }
}
