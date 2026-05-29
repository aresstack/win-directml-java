package com.aresstack.windirectml.windows;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * GPU-side correctness test for {@link QwenAttentionShaders} (Opt-B-Step 2).
 *
 * <p>Validates {@code rope_and_append} and {@code gqa_attention_decode} HLSL
 * shaders against a Java reference implementation on small deterministic
 * tensors (numHeads=2, kvHeads=1, headDim=8, maxSeqLen=4). If this test
 * passes, the shaders are numerically correct and any later token-sequence
 * divergence in end-to-end decode must be a wiring bug, not a shader bug.
 *
 * <p>Skipped by default. Run manually on a Windows machine with a working
 * DirectML adapter:
 * <pre>
 *   gradlew.bat :directml-windows-bindings:test
 *      --tests com.aresstack.windirectml.windows.QwenAttentionShadersGpuTest
 *      -Dqwen.gpu.test=true
 * </pre>
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "qwen.gpu.test", matches = "true")
class QwenAttentionShadersGpuTest {

    // Small test geometry — small enough for a hand-traceable reference,
    // large enough to exercise tree-reduce of partial dots (headDim=8 still
    // goes through 4-level reduction in the shader).
    private static final int NUM_HEADS = 2;
    private static final int KV_HEADS = 1;
    private static final int HEAD_DIM = 8;
    private static final int MAX_SEQ = 4;
    private static final int Q_SIZE = NUM_HEADS * HEAD_DIM;          // 16
    private static final int KV_SIZE = KV_HEADS * HEAD_DIM;          // 8
    private static final int CACHE_SIZE = KV_HEADS * MAX_SEQ * HEAD_DIM; // 32
    private static final float ROPE_THETA = 10000.0f;
    private static final int Q_HEADS_PER_KV = NUM_HEADS / KV_HEADS;       // 2

    private static final float TOL = 1e-4f;

    private static WindowsBindings wb;

    @BeforeAll
    static void initGpu() throws WindowsNativeException {
        wb = new WindowsBindings();
        wb.init("directml");
    }

    @AfterAll
    static void closeGpu() {
        if (wb != null) wb.close();
    }

    @Test
    void ropeAndAppend_matchesCpuReference() throws WindowsNativeException {
        try (Arena a = Arena.ofShared()) {
            var dev = wb.getD3d12Device();
            var queue = wb.getCommandQueue();

            // ── 1. Generate deterministic test data ─────────────────
            Random rng = new Random(42);
            float[] q = randomTensor(rng, Q_SIZE);
            float[] k = randomTensor(rng, KV_SIZE);
            float[] v = randomTensor(rng, KV_SIZE);
            int pos = 2; // write into KCache/VCache at position 2

            // ── 2. Allocate GPU buffers ─────────────────────────────
            var qBuf = D3D12Bindings.createDefaultBuffer(dev, Q_SIZE * 4L, a);
            var kBuf = D3D12Bindings.createDefaultBuffer(dev, KV_SIZE * 4L, a);
            var vBuf = D3D12Bindings.createDefaultBuffer(dev, KV_SIZE * 4L, a);
            var kCacheBuf = D3D12Bindings.createDefaultBuffer(dev, CACHE_SIZE * 4L, a);
            var vCacheBuf = D3D12Bindings.createDefaultBuffer(dev, CACHE_SIZE * 4L, a);

            try {
                D3D12Bindings.uploadFloats(dev, queue, qBuf, q, a);
                D3D12Bindings.uploadFloats(dev, queue, kBuf, k, a);
                D3D12Bindings.uploadFloats(dev, queue, vBuf, v, a);
                // KCache / VCache zero-initialised so the assertion on the tail
                // is meaningful (we want to be sure the shader only writes at pos).
                D3D12Bindings.uploadFloats(dev, queue, kCacheBuf, new float[CACHE_SIZE], a);
                D3D12Bindings.uploadFloats(dev, queue, vCacheBuf, new float[CACHE_SIZE], a);

                // ── 3. Compile shader ───────────────────────────────
                var allocator = D3D12Bindings.createCommandAllocator(
                        dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                var cmdList = D3D12Bindings.createCommandList(
                        dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                GpuComputeKernel rope = new GpuComputeKernel(wb, cmdList,
                        QwenAttentionShaders.ROPE_AND_APPEND_HLSL,
                        "qwen_rope_and_append_test",
                        5, 6, 32);

                // ── 4. Record + dispatch ────────────────────────────
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(qBuf),
                        D3D12Bindings.getGpuVirtualAddress(kBuf),
                        D3D12Bindings.getGpuVirtualAddress(vBuf),
                        D3D12Bindings.getGpuVirtualAddress(kCacheBuf),
                        D3D12Bindings.getGpuVirtualAddress(vCacheBuf)
                };
                int[] constants = {
                        NUM_HEADS, KV_HEADS, HEAD_DIM, pos, MAX_SEQ,
                        Float.floatToRawIntBits(ROPE_THETA)
                };
                int elementCount = (NUM_HEADS + KV_HEADS) * 32; // groups * groupSize
                rope.recordDispatch(cmdList, uavs, constants, elementCount);
                D3D12Bindings.executeAndWait(dev, queue, cmdList, a);
                rope.close();

                // ── 5. Read back results ────────────────────────────
                float[] qGpu = D3D12Bindings.readbackFloats(dev, queue, qBuf, Q_SIZE, a);
                float[] kCacheGpu = D3D12Bindings.readbackFloats(dev, queue, kCacheBuf, CACHE_SIZE, a);
                float[] vCacheGpu = D3D12Bindings.readbackFloats(dev, queue, vCacheBuf, CACHE_SIZE, a);

                // ── 6. CPU reference ────────────────────────────────
                float[] qExpected = q.clone();
                for (int h = 0; h < NUM_HEADS; h++) {
                    applyRoPECpu(qExpected, h * HEAD_DIM, HEAD_DIM, pos, ROPE_THETA);
                }
                float[] kRotatedExpected = k.clone();
                for (int h = 0; h < KV_HEADS; h++) {
                    applyRoPECpu(kRotatedExpected, h * HEAD_DIM, HEAD_DIM, pos, ROPE_THETA);
                }

                // ── 7. Assertions ───────────────────────────────────
                assertArrayClose("Q rotated", qExpected, qGpu, TOL);

                // KCache: position 'pos' contains rotated K, all other positions 0
                for (int kvH = 0; kvH < KV_HEADS; kvH++) {
                    for (int p = 0; p < MAX_SEQ; p++) {
                        int base = (kvH * MAX_SEQ + p) * HEAD_DIM;
                        for (int d = 0; d < HEAD_DIM; d++) {
                            float got = kCacheGpu[base + d];
                            float want = (p == pos) ? kRotatedExpected[kvH * HEAD_DIM + d] : 0.0f;
                            assertEquals(want, got, TOL,
                                    "KCache[kvH=" + kvH + ", p=" + p + ", d=" + d + "]");
                        }
                    }
                }
                // VCache: position 'pos' contains untouched V, all other positions 0
                for (int kvH = 0; kvH < KV_HEADS; kvH++) {
                    for (int p = 0; p < MAX_SEQ; p++) {
                        int base = (kvH * MAX_SEQ + p) * HEAD_DIM;
                        for (int d = 0; d < HEAD_DIM; d++) {
                            float got = vCacheGpu[base + d];
                            float want = (p == pos) ? v[kvH * HEAD_DIM + d] : 0.0f;
                            assertEquals(want, got, TOL,
                                    "VCache[kvH=" + kvH + ", p=" + p + ", d=" + d + "]");
                        }
                    }
                }
            } finally {
                DxgiBindings.release(qBuf);
                DxgiBindings.release(kBuf);
                DxgiBindings.release(vBuf);
                DxgiBindings.release(kCacheBuf);
                DxgiBindings.release(vCacheBuf);
            }
        }
    }

    @Test
    void gqaAttentionDecode_matchesCpuReference() throws WindowsNativeException {
        try (Arena a = Arena.ofShared()) {
            var dev = wb.getD3d12Device();
            var queue = wb.getCommandQueue();

            Random rng = new Random(123);
            int seqLen = 3;

            // ── 1. Build a fully populated KCache / VCache for seqLen=3
            //      (positions 0..2 valid, position 3 garbage/zero).
            float[] q = randomTensor(rng, Q_SIZE);
            float[] kCache = new float[CACHE_SIZE];
            float[] vCache = new float[CACHE_SIZE];
            for (int kvH = 0; kvH < KV_HEADS; kvH++) {
                for (int p = 0; p < seqLen; p++) {
                    int base = (kvH * MAX_SEQ + p) * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        kCache[base + d] = rng.nextFloat() * 2 - 1;
                        vCache[base + d] = rng.nextFloat() * 2 - 1;
                    }
                }
                // Position seqLen..MAX_SEQ-1 stays 0 (must not be read by shader)
            }

            // ── 2. Allocate + upload ────────────────────────────────
            var qBuf = D3D12Bindings.createDefaultBuffer(dev, Q_SIZE * 4L, a);
            var kCacheBuf = D3D12Bindings.createDefaultBuffer(dev, CACHE_SIZE * 4L, a);
            var vCacheBuf = D3D12Bindings.createDefaultBuffer(dev, CACHE_SIZE * 4L, a);
            var attnOutBuf = D3D12Bindings.createDefaultBuffer(dev, Q_SIZE * 4L, a);

            try {
                D3D12Bindings.uploadFloats(dev, queue, qBuf, q, a);
                D3D12Bindings.uploadFloats(dev, queue, kCacheBuf, kCache, a);
                D3D12Bindings.uploadFloats(dev, queue, vCacheBuf, vCache, a);
                D3D12Bindings.uploadFloats(dev, queue, attnOutBuf, new float[Q_SIZE], a);

                // ── 3. Compile + dispatch ────────────────────────────
                var allocator = D3D12Bindings.createCommandAllocator(
                        dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                var cmdList = D3D12Bindings.createCommandList(
                        dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                GpuComputeKernel attn = new GpuComputeKernel(wb, cmdList,
                        QwenAttentionShaders.GQA_ATTENTION_DECODE_HLSL,
                        "qwen_gqa_attention_decode_test",
                        4, 7, 64);

                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(qBuf),
                        D3D12Bindings.getGpuVirtualAddress(kCacheBuf),
                        D3D12Bindings.getGpuVirtualAddress(vCacheBuf),
                        D3D12Bindings.getGpuVirtualAddress(attnOutBuf)
                };
                float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
                int[] constants = {
                        NUM_HEADS, KV_HEADS, HEAD_DIM, Q_HEADS_PER_KV,
                        seqLen, MAX_SEQ,
                        Float.floatToRawIntBits(scale)
                };
                int elementCount = NUM_HEADS * 64; // numHeads groups * groupSize 64
                attn.recordDispatch(cmdList, uavs, constants, elementCount);
                D3D12Bindings.executeAndWait(dev, queue, cmdList, a);
                attn.close();

                float[] attnGpu = D3D12Bindings.readbackFloats(dev, queue, attnOutBuf, Q_SIZE, a);

                // ── 4. CPU reference: same algorithm as Qwen2Runtime.processLayerDecode
                float[] attnExpected = new float[Q_SIZE];
                for (int h = 0; h < NUM_HEADS; h++) {
                    int kvH = h / Q_HEADS_PER_KV;
                    int qOff = h * HEAD_DIM;
                    int outOff = h * HEAD_DIM;
                    float[] scores = new float[seqLen];
                    for (int p = 0; p < seqLen; p++) {
                        int kOff = (kvH * MAX_SEQ + p) * HEAD_DIM;
                        float s = 0;
                        for (int d = 0; d < HEAD_DIM; d++) {
                            s += q[qOff + d] * kCache[kOff + d];
                        }
                        scores[p] = s * scale;
                    }
                    softmaxCpu(scores);
                    for (int p = 0; p < seqLen; p++) {
                        int vOff = (kvH * MAX_SEQ + p) * HEAD_DIM;
                        float w = scores[p];
                        for (int d = 0; d < HEAD_DIM; d++) {
                            attnExpected[outOff + d] += w * vCache[vOff + d];
                        }
                    }
                }

                assertArrayClose("AttnOut", attnExpected, attnGpu, TOL);
            } finally {
                DxgiBindings.release(qBuf);
                DxgiBindings.release(kCacheBuf);
                DxgiBindings.release(vCacheBuf);
                DxgiBindings.release(attnOutBuf);
            }
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    private static float[] randomTensor(Random rng, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) x[i] = rng.nextFloat() * 2 - 1;
        return x;
    }

    /**
     * GPT-J-style RoPE — pairs (x[i], x[halfDim+i]). Same as Qwen2Runtime.applyRoPE.
     */
    private static void applyRoPECpu(float[] vec, int offset, int dim, int pos, float theta) {
        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++) {
            double freq = 1.0 / Math.pow(theta, (2.0 * i) / dim);
            double angle = pos * freq;
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            float x0 = vec[offset + i];
            float x1 = vec[offset + halfDim + i];
            vec[offset + i] = x0 * cos - x1 * sin;
            vec[offset + halfDim + i] = x0 * sin + x1 * cos;
        }
    }

    private static void softmaxCpu(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) if (v > max) max = v;
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) Math.exp(x[i] - max);
            sum += x[i];
        }
        for (int i = 0; i < x.length; i++) x[i] /= sum;
    }

    private static void assertArrayClose(String label, float[] want, float[] got, float tol) {
        if (want.length != got.length) {
            fail(label + ": length mismatch want=" + want.length + " got=" + got.length);
        }
        for (int i = 0; i < want.length; i++) {
            float diff = Math.abs(want[i] - got[i]);
            if (diff > tol) {
                fail(String.format("%s[%d] mismatch: want=%.6f got=%.6f diff=%.6e (tol=%.6e)",
                        label, i, want[i], got[i], diff, tol));
            }
        }
    }
}
