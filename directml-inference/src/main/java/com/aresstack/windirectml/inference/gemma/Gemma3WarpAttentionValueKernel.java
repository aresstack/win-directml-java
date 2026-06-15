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
 * WARP/DirectML attention value-aggregation kernel for Gemma 3 (GEMMA-WARP-8a).
 *
 * <p>{@code context[head, c] = sum_j prob[head, j] * V[j][kvHead][c]} with GQA mapping
 * {@code kvHead = head / groupsPerKv} — the weighted sum of value vectors that follows the softmax,
 * mirroring the {@code attnOut += a * v} step of {@code Gemma3ReferenceForwardPass}. One thread per
 * {@code (head, c)}. The HLSL is compiled once; each call uploads, dispatches and reads back. Requires
 * {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpAttentionValueKernel implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpAttentionValueKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.device = windowsBindings.getD3d12Device();
        this.queue = windowsBindings.getCommandQueue();
        this.arena = Arena.ofShared();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpAttention.ATTENTION_VALUE_HLSL, "gemma3_attention_value",
                3, 5, Gemma3WarpAttention.GROUP_SIZE);
    }

    /**
     * Aggregate values by attention probabilities for one query position.
     *
     * @param prob       attention probabilities {@code [numHeads * cols]} (head {@code h} row at {@code h*cols})
     * @param values     values {@code [cols * (numKvHeads*headDim)]} (value {@code j} head {@code kv} at
     *                   {@code j*kvDim + kv*headDim})
     * @param numHeads   query heads
     * @param numKvHeads kv heads
     * @param headDim    head width
     * @param cols       number of attended key/value positions
     * @return context {@code [numHeads * headDim]} (head {@code h} at {@code h*headDim})
     */
    public float[] aggregate(float[] prob, float[] values, int numHeads, int numKvHeads, int headDim, int cols)
            throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(prob, "prob");
        Objects.requireNonNull(values, "values");
        if (numHeads < 1 || numKvHeads < 1 || headDim < 1 || cols < 1) {
            throw new IllegalArgumentException("numHeads/numKvHeads/headDim/cols must be positive");
        }
        if (numHeads % numKvHeads != 0) {
            throw new IllegalArgumentException("numHeads must be a multiple of numKvHeads");
        }
        int kvDim = numKvHeads * headDim;
        if (prob.length != (long) numHeads * cols) {
            throw new IllegalArgumentException("prob length must equal numHeads*cols: prob=" + prob.length);
        }
        if (values.length != (long) cols * kvDim) {
            throw new IllegalArgumentException("values length must equal cols*kvDim: values=" + values.length
                    + ", expected=" + ((long) cols * kvDim));
        }
        int groupsPerKv = numHeads / numKvHeads;
        int total = numHeads * headDim;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment pBuf = D3D12Bindings.createDefaultBuffer(device, prob.length * 4L, a);
            MemorySegment vBuf = D3D12Bindings.createDefaultBuffer(device, values.length * 4L, a);
            MemorySegment cBuf = D3D12Bindings.createDefaultBuffer(device, total * 4L, a);
            try {
                D3D12Bindings.uploadFloats(device, queue, pBuf, prob, a);
                D3D12Bindings.uploadFloats(device, queue, vBuf, values, a);
                D3D12Bindings.uploadFloats(device, queue, cBuf, new float[total], a);

                MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                MemorySegment cmdList = D3D12Bindings.createCommandList(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(pBuf),
                        D3D12Bindings.getGpuVirtualAddress(vBuf),
                        D3D12Bindings.getGpuVirtualAddress(cBuf)
                };
                int[] constants = {numHeads, groupsPerKv, headDim, kvDim, cols};
                kernel.recordDispatch(cmdList, uavs, constants, total);
                D3D12Bindings.executeAndWait(device, queue, cmdList, a);

                return D3D12Bindings.readbackFloats(device, queue, cBuf, total, a);
            } finally {
                DxgiBindings.release(pBuf);
                DxgiBindings.release(vBuf);
                DxgiBindings.release(cBuf);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpAttentionValueKernel is closed");
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
