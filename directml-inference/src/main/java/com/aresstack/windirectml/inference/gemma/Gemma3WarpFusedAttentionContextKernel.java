package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.GpuComputeKernel;
import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Fused single-query attention context for Gemma 3 (GEMMA-WARP-14b): one dispatch computes
 * {@code context = softmax(scale · q·K_visible) · V_visible}, replacing the staged
 * {@link Gemma3WarpAttentionScoresKernel} → {@link Gemma3WarpSoftmaxKernel} →
 * {@link Gemma3WarpAttentionValueKernel} (3 dispatches + 2 inter-dispatch UAV barriers → 1 dispatch).
 *
 * <p>Flash-attention-style online softmax (running max/sum + rescaled value accumulator), so no
 * {@code [seqLen]} scores buffer is materialised. One thread group per attention head; thread {@code c}
 * owns output dim {@code c} and the per-key dot-product is a group reduction. Same Gemma layout as the
 * staged path: GQA ({@code kvHead = head / groupsPerKv}), causal + sliding-window via the visible key range
 * {@code [firstValid, queryPos]}, explicit {@code head_dim} (≤ group size 256). Numerically equal to the
 * staged path within fp32 tolerance (different accumulation order). Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpFusedAttentionContextKernel implements AutoCloseable {

    /** Group size = max supported head_dim. Gemma 3 head_dim is 256. */
    public static final int GROUP_SIZE = 256;

    private static final String FUSED_ATTENTION_HLSL = """
            RWByteAddressBuffer Q   : register(u0);
            RWByteAddressBuffer K   : register(u1);
            RWByteAddressBuffer V   : register(u2);
            RWByteAddressBuffer Ctx : register(u3);
            cbuffer CB : register(b0) {
                uint numHeads; uint groupsPerKv; uint headDim; uint kvDim;
                uint firstValid; uint queryPos; uint scale_bits;
            };

            groupshared float red[256];

            [numthreads(256, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
                uint head = gid.x;
                if (head >= numHeads) {
                    return;
                }
                uint c = gi;
                bool active = c < headDim;
                uint kvHead = head / groupsPerKv;
                uint qBase = head * headDim;
                float scale = asfloat(scale_bits);
                float qc = active ? asfloat(Q.Load((qBase + c) * 4)) : 0.0f;

                float m = -1e30f;   // running max
                float l = 0.0f;     // running sum of exp
                float acc = 0.0f;   // running sum of p*V for this output dim c

                for (uint j = firstValid; j <= queryPos; j++) {
                    uint kBase = j * kvDim + kvHead * headDim;
                    red[c] = active ? qc * asfloat(K.Load((kBase + c) * 4)) : 0.0f;
                    GroupMemoryBarrierWithGroupSync();
                    for (uint st = 128; st > 0; st >>= 1) {
                        if (c < st) {
                            red[c] += red[c + st];
                        }
                        GroupMemoryBarrierWithGroupSync();
                    }
                    float s = red[0] * scale;            // q·K[j] scaled (all threads read red[0])
                    GroupMemoryBarrierWithGroupSync();   // finish reads before the next iter overwrites red

                    float mNew = max(m, s);
                    float corr = exp(m - mNew);
                    float p = exp(s - mNew);
                    l = l * corr + p;
                    acc = acc * corr + (active ? p * asfloat(V.Load((kBase + c) * 4)) : 0.0f);
                    m = mNew;
                }

                if (active) {
                    Ctx.Store((qBase + c) * 4, asuint(acc / l));
                }
            }
            """;

    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpFusedAttentionContextKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.arena = Arena.ofShared();
        MemorySegment device = windowsBindings.getD3d12Device();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                FUSED_ATTENTION_HLSL, "gemma3_fused_attention", 4, 7, GROUP_SIZE);
    }

    /**
     * Fused attention context for one query at {@code queryPos} attending the visible key range
     * {@code [firstValid, queryPos]} of the resident K/V cache. {@code keys}/{@code values} may be larger
     * than the visible range (the GPU-resident KV cache); only {@code [0, queryPos]} is read.
     *
     * @return resident context buffer of length {@code numHeads * headDim}
     */
    public WarpGpuBuffer context(WarpExecutionContext ctx, WarpGpuBuffer q, WarpGpuBuffer keys,
                                 WarpGpuBuffer values, int numHeads, int numKvHeads, int headDim, int kvDim,
                                 int queryPos, int firstValid, float scale) throws WindowsNativeException {
        ensureOpen();
        if (numHeads % numKvHeads != 0) {
            throw new IllegalArgumentException("numHeads must be a multiple of numKvHeads");
        }
        if (headDim > GROUP_SIZE) {
            throw new IllegalArgumentException("headDim " + headDim + " exceeds fused-attention group size " + GROUP_SIZE);
        }
        if (q.elementCount() != numHeads * headDim) {
            throw new IllegalArgumentException("q length mismatch: " + q.elementCount() + " != " + (numHeads * headDim));
        }
        long visible = (long) (queryPos + 1) * kvDim;
        if (keys.elementCount() < visible || values.elementCount() < visible) {
            throw new IllegalArgumentException("keys/values too small for queryPos " + queryPos);
        }
        int groupsPerKv = numHeads / numKvHeads;
        WarpGpuBuffer out = ctx.allocate(numHeads * headDim);
        ctx.dispatch(kernel,
                new long[]{q.gpuAddress(), keys.gpuAddress(), values.gpuAddress(), out.gpuAddress()},
                new int[]{numHeads, groupsPerKv, headDim, kvDim, firstValid, queryPos,
                        Float.floatToRawIntBits(scale)},
                numHeads * GROUP_SIZE); // -> numHeads thread groups
        return out;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpFusedAttentionContextKernel is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            kernel.close();
            arena.close();
        }
    }
}
