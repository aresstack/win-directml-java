package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.DxgiBindings;
import com.aresstack.windirectml.windows.GpuComputeKernel;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * WARP/DirectML attention-scores kernel for Gemma 3 (GEMMA-WARP-7b).
 *
 * <p>The GPU mirror of the scaled {@code QK^T} step in {@code Gemma3ReferenceForwardPass}: for one
 * query position it produces {@code scores[head, j] = attentionScale * dot(Q[head], K[j][kvHead])} for
 * every visible key {@code j in [firstValid, queryPos]} (GQA mapping {@code kvHead = head/groupsPerKv}),
 * and a large-negative sentinel ({@link Gemma3WarpAttention#SCORE_SENTINEL}) for masked positions. The
 * visible range comes from {@link Gemma3AttentionLayout} (full layers → {@code firstValid=0}; local
 * layers → the sliding window), so this kernel applies Gemma's local/global + causal mask without
 * imitating any other family's attention. The HLSL is compiled once; each call uploads, dispatches and
 * reads back. Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpAttentionScoresKernel implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpAttentionScoresKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.device = windowsBindings.getD3d12Device();
        this.queue = windowsBindings.getCommandQueue();
        this.arena = Arena.ofShared();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpAttention.ATTENTION_SCORES_HLSL, "gemma3_attention_scores",
                3, 8, Gemma3WarpAttention.GROUP_SIZE);
    }

    /**
     * Compute the masked, scaled attention scores for query position {@code queryPos}.
     *
     * @param q            RoPE'd query of length {@code numHeads * headDim} (head {@code h} at {@code h*headDim})
     * @param keys         RoPE'd keys of length {@code seqLen * (numKvHeads*headDim)} (key {@code j} head
     *                     {@code kv} at {@code j*kvDim + kv*headDim}, {@code kvDim = numKvHeads*headDim})
     * @param numHeads     query heads
     * @param numKvHeads   kv heads ({@code numHeads % numKvHeads == 0})
     * @param headDim      head width (Gemma: 256; never {@code hidden/heads})
     * @param seqLen       number of cached keys
     * @param queryPos     position of the query (causal upper bound, inclusive)
     * @param firstValid   first visible key (from {@link Gemma3AttentionLayout#firstValidKey})
     * @param scale        attention scale ({@link Gemma3AttentionLayout#attentionScale})
     * @return scores of length {@code numHeads * seqLen} (head {@code h} row at {@code h*seqLen})
     */
    public float[] scores(float[] q, float[] keys, int numHeads, int numKvHeads, int headDim,
                          int seqLen, int queryPos, int firstValid, float scale)
            throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(keys, "keys");
        if (numHeads < 1 || numKvHeads < 1 || headDim < 1 || seqLen < 1) {
            throw new IllegalArgumentException("numHeads/numKvHeads/headDim/seqLen must be positive");
        }
        if (numHeads % numKvHeads != 0) {
            throw new IllegalArgumentException("numHeads must be a multiple of numKvHeads: numHeads="
                    + numHeads + ", numKvHeads=" + numKvHeads);
        }
        int kvDim = numKvHeads * headDim;
        if (q.length != (long) numHeads * headDim) {
            throw new IllegalArgumentException("q length must equal numHeads*headDim: q=" + q.length);
        }
        if (keys.length != (long) seqLen * kvDim) {
            throw new IllegalArgumentException("keys length must equal seqLen*kvDim: keys=" + keys.length
                    + ", expected=" + ((long) seqLen * kvDim));
        }
        if (queryPos < 0 || queryPos >= seqLen) {
            throw new IllegalArgumentException("queryPos out of range: " + queryPos + " (seqLen=" + seqLen + ")");
        }
        int groupsPerKv = numHeads / numKvHeads;
        int total = numHeads * seqLen;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment qBuf = D3D12Bindings.createDefaultBuffer(device, q.length * 4L, a);
            MemorySegment kBuf = D3D12Bindings.createDefaultBuffer(device, keys.length * 4L, a);
            MemorySegment sBuf = D3D12Bindings.createDefaultBuffer(device, total * 4L, a);
            try {
                D3D12Bindings.uploadFloats(device, queue, qBuf, q, a);
                D3D12Bindings.uploadFloats(device, queue, kBuf, keys, a);
                D3D12Bindings.uploadFloats(device, queue, sBuf, new float[total], a);

                MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                MemorySegment cmdList = D3D12Bindings.createCommandList(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(qBuf),
                        D3D12Bindings.getGpuVirtualAddress(kBuf),
                        D3D12Bindings.getGpuVirtualAddress(sBuf)
                };
                int[] constants = {numHeads, groupsPerKv, headDim, kvDim,
                        seqLen, queryPos, firstValid, Float.floatToRawIntBits(scale)};
                kernel.recordDispatch(cmdList, uavs, constants, total);
                D3D12Bindings.executeAndWait(device, queue, cmdList, a);

                return D3D12Bindings.readbackFloats(device, queue, sBuf, total, a);
            } finally {
                DxgiBindings.release(qBuf);
                DxgiBindings.release(kBuf);
                DxgiBindings.release(sBuf);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpAttentionScoresKernel is closed");
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
