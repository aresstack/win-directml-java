package com.aresstack.windirectml.runtime;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * End-to-end roundtrip test for {@link DirectMlContextImpl} + {@link DefaultGpuBuffer}.
 * <p>
 * Validates the resource-state contract before the first real DirectML kernel
 * (GEMM / Linear) lands:
 * <pre>
 *   CpuTensor(float[])
 *     → ctx.allocateBufferFor(...)
 *     → buffer.upload(...)       // COMMON → COPY_DEST → UAV
 *     → buffer.download(...)     // UAV    → COPY_SOURCE → UAV
 *     → assert same floats
 * </pre>
 * The test requires a working DXGI / D3D12 / DirectML stack. On platforms where
 * initialisation fails (non-Windows CI, headless dev box, missing GPU adapter),
 * it skips via {@link org.junit.jupiter.api.Assumptions#assumeTrue}.
 */
class DirectMlContextRoundtripTest {

    @Test
    void uploadDownloadRoundtripPreservesFloats() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Roundtrip test requires Windows + D3D12");

        DirectMlContextImpl ctx = new DirectMlContextImpl("auto");
        try {
            try {
                ctx.initialize();
            } catch (DirectMlRuntimeException e) {
                // No GPU adapter / no D3D12 device on this box – treat as skip,
                // not as failure. CI / dev boxes without a discrete or WARP
                // adapter would otherwise burn red without telling us anything.
                Throwable root = e.getCause();
                assumeTrue(false,
                        "Skipping: D3D12 stack init failed: "
                                + (root != null ? root.getMessage() : e.getMessage()));
                return; // unreachable, keeps the compiler happy
            }
            assumeTrue(ctx.isReady(), "Context did not become ready");

            // 384 floats is the MiniLM embedding width – realistic size, still tiny.
            final int n = 384;
            float[] expected = new float[n];
            for (int i = 0; i < n; i++) {
                expected[i] = (float) Math.sin(i * 0.017) + 0.001f * i;
            }

            CpuTensor input = CpuTensor.float32(TensorShape.of(1, n), expected);

            try (GpuBuffer buf = ctx.allocateBufferFor(input, GpuBuffer.BufferUsage.ACTIVATION)) {
                assertNotNull(buf);
                assertEquals((long) n * Float.BYTES, buf.sizeInBytes(), "buffer size");

                buf.upload(input);

                ByteBuffer downloadStorage = ByteBuffer
                        .allocateDirect(n * Float.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN);
                CpuTensor output = new CpuTensor(
                        TensorShape.of(1, n),
                        TensorLayout.rowMajor(TensorShape.of(1, n)),
                        TensorDataType.FLOAT32,
                        downloadStorage);

                buf.download(output);

                float[] actual = new float[n];
                FloatBuffer fv = downloadStorage.duplicate()
                        .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                fv.position(0);
                fv.get(actual, 0, n);

                assertArrayEquals(expected, actual, 0.0f,
                        "GPU roundtrip must be bit-exact for FLOAT32 buffers");
            }
        } finally {
            ctx.close();
        }
    }

    @Test
    void repeatedRoundtripsKeepBufferInUavSteadyState() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Roundtrip test requires Windows + D3D12");

        DirectMlContextImpl ctx = new DirectMlContextImpl("auto");
        try {
            try {
                ctx.initialize();
            } catch (DirectMlRuntimeException e) {
                assumeTrue(false, "Skipping: D3D12 stack init failed: " + e.getMessage());
                return;
            }
            assumeTrue(ctx.isReady(), "Context did not become ready");

            final int n = 16;
            float[] a = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
            float[] b = new float[]{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16};

            CpuTensor inA = CpuTensor.float32(TensorShape.of(n), a);
            CpuTensor inB = CpuTensor.float32(TensorShape.of(n), b);

            try (GpuBuffer buf = ctx.allocateBufferFor(inA, GpuBuffer.BufferUsage.ACTIVATION)) {
                // First cycle: COMMON → UAV
                buf.upload(inA);
                CpuTensor out1 = allocCpu(n);
                buf.download(out1);
                assertArrayEquals(a, readFloats(out1), 0.0f);

                // Second cycle: UAV → UAV (this is what would have tripped the
                // debug layer with the old, implicit-promotion upload path).
                buf.upload(inB);
                CpuTensor out2 = allocCpu(n);
                buf.download(out2);
                assertArrayEquals(b, readFloats(out2), 0.0f);

                // Third cycle: download again without re-upload.
                CpuTensor out3 = allocCpu(n);
                buf.download(out3);
                assertArrayEquals(b, readFloats(out3), 0.0f,
                        "Download without re-upload must return the last uploaded data");
            }
        } finally {
            ctx.close();
        }
    }

    private static CpuTensor allocCpu(int n) {
        ByteBuffer storage = ByteBuffer.allocateDirect(n * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        TensorShape shape = TensorShape.of(n);
        return new CpuTensor(shape, TensorLayout.rowMajor(shape), TensorDataType.FLOAT32, storage);
    }

    private static float[] readFloats(CpuTensor t) {
        int n = (int) t.shape().elementCount();
        float[] out = new float[n];
        FloatBuffer fv = t.data().duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        fv.position(0);
        fv.get(out, 0, n);
        return out;
    }
}

