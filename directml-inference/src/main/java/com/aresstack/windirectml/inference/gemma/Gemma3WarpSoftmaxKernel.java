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
 * WARP/DirectML row-wise softmax kernel for Gemma 3 attention (GEMMA-WARP-8a).
 *
 * <p>Numerically-stable softmax (max-subtract → exp → normalize) over each row of a
 * {@code [numRows * cols]} score matrix; one D3D12 thread group reduces one row. Masked entries carrying
 * {@link Gemma3WarpAttention#SCORE_SENTINEL} exp to ~0, matching the reference which simply skips them.
 * Mirror of the softmax in {@code Gemma3ReferenceForwardPass} (note: the reference accumulates in
 * {@code double}, this kernel in {@code float} — see the documented single-layer tolerance). Requires
 * {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpSoftmaxKernel implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpSoftmaxKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.device = windowsBindings.getD3d12Device();
        this.queue = windowsBindings.getCommandQueue();
        this.arena = Arena.ofShared();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpAttention.SOFTMAX_HLSL, "gemma3_softmax",
                2, 2, Gemma3WarpAttention.GROUP_SIZE);
    }

    /** Row-wise softmax of a {@code [numRows * cols]} matrix, allocating the output. */
    public float[] softmaxRows(float[] scores, int numRows, int cols) throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(scores, "scores");
        if (numRows < 1 || cols < 1) {
            throw new IllegalArgumentException("numRows and cols must be positive: numRows="
                    + numRows + ", cols=" + cols);
        }
        if (scores.length != (long) numRows * cols) {
            throw new IllegalArgumentException("scores length must equal numRows*cols: scores="
                    + scores.length + ", expected=" + ((long) numRows * cols));
        }
        int total = numRows * cols;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment inBuf = D3D12Bindings.createDefaultBuffer(device, total * 4L, a);
            MemorySegment outBuf = D3D12Bindings.createDefaultBuffer(device, total * 4L, a);
            try {
                D3D12Bindings.uploadFloats(device, queue, inBuf, scores, a);
                D3D12Bindings.uploadFloats(device, queue, outBuf, new float[total], a);

                MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                MemorySegment cmdList = D3D12Bindings.createCommandList(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(inBuf),
                        D3D12Bindings.getGpuVirtualAddress(outBuf)
                };
                int[] constants = {numRows, cols};
                // one group per row
                kernel.recordDispatch(cmdList, uavs, constants, numRows * Gemma3WarpAttention.GROUP_SIZE);
                D3D12Bindings.executeAndWait(device, queue, cmdList, a);

                return D3D12Bindings.readbackFloats(device, queue, outBuf, total, a);
            } finally {
                DxgiBindings.release(inBuf);
                DxgiBindings.release(outBuf);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpSoftmaxKernel is closed");
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
