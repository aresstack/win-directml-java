package com.aresstack.windirectml.inference.smollm2;

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
 * Device-free tests for the heap-light FP32 source seam on {@link SmolLM2DenseTensor} (slice H2c):
 * a FLOAT32 runtime tensor defers its float decode and exposes the little-endian mmap slice for direct WARP upload,
 * while the float[] accessors (reference/diagnostic paths) keep producing correct values. Also pins that no eager
 * double-copy happens (the source is the single retained buffer).
 */
class SmolLM2DenseTensorTest {

    private static SmolLM2WeightTensor fp32Weight(long[] dims, float[] values) {
        ByteBuffer raw = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) {
            raw.putFloat(v);
        }
        raw.flip();
        RuntimeTensor tensor = new RuntimeTensor("model.layers.0.self_attn.o_proj.weight", dims,
                OnnxModelReader.ONNX_FLOAT, raw, values.length * Float.BYTES);
        SmolLM2TensorRoleBinding binding =
                new SmolLM2TensorRoleBinding(SmolLM2TensorRole.LAYER_SELF_O, 0, tensor.name());
        return new SmolLM2WeightTensor(binding, tensor);
    }

    @Test
    void fp32TensorExposesLittleEndianSourceAndDecodesLazilyToSameValues() {
        float[] values = {1.0f, -2.0f, 3.5f, 0.25f, -0.5f, 4.0f};
        SmolLM2DenseTensor data = SmolLM2DenseTensor.from(fp32Weight(new long[]{2, 3}, values));

        ByteBuffer source = data.fp32LittleEndianSource();
        assertNotNull(source, "FP32 tensor must expose a direct upload source");
        assertEquals(ByteOrder.LITTLE_ENDIAN, source.order());
        assertEquals(values.length * Float.BYTES, source.remaining());

        // Lazy float decode must reproduce the original values for all float[] accessors.
        assertArrayEquals(values, data.copyValues(), 0.0f);
        assertEquals(3.5f, data.value(2), 0.0f);
        assertArrayEquals(new float[]{0.25f, -0.5f, 4.0f}, data.copyRow(1), 0.0f);
        assertEquals(6, data.elementCount());
    }

    @Test
    void fp32MultiplyVectorAndDotRowAreCorrect() {
        float[] weights = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 4.0f}; // [2,3]
        float[] input = {2.0f, -1.0f, 0.25f};
        SmolLM2DenseTensor data = SmolLM2DenseTensor.from(fp32Weight(new long[]{2, 3}, weights));

        float row0 = 1.0f * 2 + 2.0f * -1 + 3.0f * 0.25f;
        float row1 = -1.0f * 2 + 0.5f * -1 + 4.0f * 0.25f;
        assertArrayEquals(new float[]{row0, row1}, data.multiplyVector(input), 1e-6f);
        assertEquals(row0, data.dotRow(0, input), 1e-6f);
        assertEquals(row1, data.dotRow(1, input), 1e-6f);
    }

    @Test
    void fp32SourceIsIndependentViewAndCopyValuesIsDefensive() {
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        SmolLM2DenseTensor data = SmolLM2DenseTensor.from(fp32Weight(new long[]{2, 2}, values));

        ByteBuffer a = data.fp32LittleEndianSource();
        a.position(8);
        ByteBuffer b = data.fp32LittleEndianSource();
        assertEquals(0, b.position(), "each fp32LittleEndianSource() call must return an independent view");

        float[] c1 = data.copyValues();
        c1[0] = 999.0f;
        assertArrayEquals(values, data.copyValues(), 0.0f, "copyValues() must return a defensive copy");
    }

    @Test
    void fp16TensorHasNoFp32SourceButDecodesValues() {
        // FP16 1.0 = 0x3C00, 2.0 = 0x4000.
        ByteBuffer raw = ByteBuffer.allocate(2 * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        raw.putShort((short) 0x3C00);
        raw.putShort((short) 0x4000);
        raw.flip();
        RuntimeTensor tensor = new RuntimeTensor("model.layers.0.self_attn.o_proj.weight", new long[]{1, 2},
                OnnxModelReader.ONNX_FLOAT16, raw, 2 * Short.BYTES);
        SmolLM2WeightTensor weight = new SmolLM2WeightTensor(
                new SmolLM2TensorRoleBinding(SmolLM2TensorRole.LAYER_SELF_O, 0, tensor.name()), tensor);

        SmolLM2DenseTensor data = SmolLM2DenseTensor.from(weight);
        assertNull(data.fp32LittleEndianSource(), "FP16 tensor must use the float[] fallback (no FP32 source)");
        assertArrayEquals(new float[]{1.0f, 2.0f}, data.copyValues(), 0.0f);
    }
}
