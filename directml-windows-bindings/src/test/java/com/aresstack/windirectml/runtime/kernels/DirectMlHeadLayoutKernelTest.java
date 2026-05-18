package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.CpuTensor;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.GpuBuffer;
import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorLayout;
import com.aresstack.windirectml.runtime.TensorShape;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Verifies {@link DirectMlHeadLayoutKernel} against an explicit CPU
 * reference permutation.
 * <p>
 * Three guarantees this test enforces:
 * <ol>
 *   <li>{@code SEQ_TO_HEAD} reorders {@code [S, H, D]} into
 *       {@code [H, S, D]} bit-exactly.</li>
 *   <li>{@code HEAD_TO_SEQ} reorders {@code [H, S, D]} back into
 *       {@code [S, H, D]} bit-exactly.</li>
 *   <li>Round-trip {@code forward ∘ backward} is the identity – this is
 *       the single most important property when later wiring the kernel
 *       around {@link DirectMlAttentionKernel}.</li>
 * </ol>
 * Identity copies preserve exact float bits, so the tolerance is
 * {@code 0.0f}; any mismatch indicates a stride bug.
 */
class DirectMlHeadLayoutKernelTest {

    private static final int S = 4;
    private static final int H = 3;
    private static final int D = 8;
    private static final int N = S * H * D;

    @Test
    void seqMajorToHeadMajorMatchesCpuPermutation() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");
        DirectMlContextImpl ctx = new DirectMlContextImpl("directml");
        try {
            try { ctx.initialize(); }
            catch (DirectMlRuntimeException e) { assumeTrue(false, "no DML: " + e.getMessage()); return; }
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(), "Skipping: no DirectML device");

            Random rng = new Random(0xACE1L);
            float[] in = randomFloats(rng, N);                        // physically [S,H,D]
            float[] expected = permuteSeqToHead(in);                  // physically [H,S,D]
            float[] actual = runForward(ctx, in);

            assertArrayEquals(expected, actual, 0.0f,
                    "SEQ_TO_HEAD must produce the exact [H,S,D] permutation");
        } finally { ctx.close(); }
    }

    @Test
    void headMajorToSeqMajorMatchesCpuPermutation() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");
        DirectMlContextImpl ctx = new DirectMlContextImpl("directml");
        try {
            try { ctx.initialize(); }
            catch (DirectMlRuntimeException e) { assumeTrue(false, "no DML: " + e.getMessage()); return; }
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(), "Skipping: no DirectML device");

            Random rng = new Random(0xBEE5L);
            float[] in = randomFloats(rng, N);                        // physically [H,S,D]
            float[] expected = permuteHeadToSeq(in);                  // physically [S,H,D]
            float[] actual = runBackward(ctx, in);

            assertArrayEquals(expected, actual, 0.0f,
                    "HEAD_TO_SEQ must produce the exact [S,H,D] permutation");
        } finally { ctx.close(); }
    }

    /**
     * Critical round-trip property: applying SEQ_TO_HEAD followed by
     * HEAD_TO_SEQ on any random input must yield the original input
     * byte-for-byte. If this fails, the two stride vectors are not
     * inverse to each other and any attention pipeline that wraps the
     * kernels will silently produce garbage.
     */
    @Test
    void roundTripIsIdentity() throws DirectMlRuntimeException {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12");
        DirectMlContextImpl ctx = new DirectMlContextImpl("directml");
        try {
            try { ctx.initialize(); }
            catch (DirectMlRuntimeException e) { assumeTrue(false, "no DML: " + e.getMessage()); return; }
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(), "Skipping: no DirectML device");

            Random rng = new Random(0xC0FFEEL);
            float[] in = randomFloats(rng, N);
            float[] mid = runForward(ctx, in);
            float[] back = runBackward(ctx, mid);

            assertArrayEquals(in, back, 0.0f,
                    "SEQ_TO_HEAD ∘ HEAD_TO_SEQ must be the identity on the data");
        } finally { ctx.close(); }
    }

    // ── DirectML drivers ─────────────────────────────────────────────────

    private float[] runForward(DirectMlContextImpl ctx, float[] in) throws DirectMlRuntimeException {
        return runConversion(ctx, in, /* forward */ true);
    }

    private float[] runBackward(DirectMlContextImpl ctx, float[] in) throws DirectMlRuntimeException {
        return runConversion(ctx, in, /* forward */ false);
    }

    private float[] runConversion(DirectMlContextImpl ctx, float[] in, boolean forward)
            throws DirectMlRuntimeException {
        CpuTensor inCpu = CpuTensor.float32(TensorShape.of(N), in);
        try (GpuBuffer inBuf = ctx.allocateBufferFor(inCpu, GpuBuffer.BufferUsage.ACTIVATION);
             GpuBuffer outBuf = ctx.allocateBuffer((long) N * Float.BYTES,
                     GpuBuffer.BufferUsage.ACTIVATION);
             DirectMlHeadLayoutKernel kernel = forward
                     ? DirectMlHeadLayoutKernel.seqMajorToHeadMajor(ctx, S, H, D)
                     : DirectMlHeadLayoutKernel.headMajorToSeqMajor(ctx, S, H, D)) {

            inBuf.upload(inCpu);
            DirectMlTensor xT = tensor(inBuf, N);
            DirectMlTensor yT = tensor(outBuf, N);
            kernel.dispatch(xT, yT);

            CpuTensor outCpu = emptyCpu(N);
            outBuf.download(outCpu);
            return readFloats(outCpu, N);
        }
    }

    // ── CPU reference permutations ──────────────────────────────────────

    /** Physical [S,H,D] (input order s·H·D + h·D + d) → physical [H,S,D] (h·S·D + s·D + d). */
    private static float[] permuteSeqToHead(float[] src) {
        float[] dst = new float[N];
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < H; h++) {
                for (int d = 0; d < D; d++) {
                    int in = s * H * D + h * D + d;
                    int out = h * S * D + s * D + d;
                    dst[out] = src[in];
                }
            }
        }
        return dst;
    }

    /** Physical [H,S,D] (input order h·S·D + s·D + d) → physical [S,H,D] (s·H·D + h·D + d). */
    private static float[] permuteHeadToSeq(float[] src) {
        float[] dst = new float[N];
        for (int h = 0; h < H; h++) {
            for (int s = 0; s < S; s++) {
                for (int d = 0; d < D; d++) {
                    int in = h * S * D + s * D + d;
                    int out = s * H * D + h * D + d;
                    dst[out] = src[in];
                }
            }
        }
        return dst;
    }

    // ── small helpers ───────────────────────────────────────────────────

    private static DirectMlTensor tensor(GpuBuffer buf, int n) {
        TensorShape s = TensorShape.of(n);
        return new DirectMlTensor(s, TensorLayout.rowMajor(s), TensorDataType.FLOAT32, buf);
    }

    private static float[] randomFloats(Random rng, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) out[i] = rng.nextFloat() * 10.0f - 5.0f;
        return out;
    }

    private static CpuTensor emptyCpu(int n) {
        ByteBuffer storage = ByteBuffer.allocateDirect(n * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        TensorShape shape = TensorShape.of(n);
        return new CpuTensor(shape, TensorLayout.rowMajor(shape), TensorDataType.FLOAT32, storage);
    }

    private static float[] readFloats(CpuTensor t, int n) {
        float[] out = new float[n];
        FloatBuffer fv = t.data().duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        fv.position(0);
        fv.get(out, 0, n);
        return out;
    }
}

