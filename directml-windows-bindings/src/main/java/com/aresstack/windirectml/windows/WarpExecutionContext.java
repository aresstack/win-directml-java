package com.aresstack.windirectml.windows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Minimal GPU-resident execution seam for chaining WARP/DirectML compute kernels (GEMMA-WARP-13b-2).
 *
 * <p>Provides resident {@link WarpGpuBuffer} factories and a {@link #dispatch} that records one compute
 * kernel into a command list and submits it (one submit + fence wait, counted by
 * {@link WarpSubmissionStats}) <b>without any CPU readback</b> — the output stays in a GPU buffer for the
 * next kernel. This removes the per-kernel upload→dispatch→readback→re-upload round-trips of the
 * float[] API; the only readback is the caller's final {@link WarpGpuBuffer#readback()}.</p>
 *
 * <p>Each {@code dispatch} is still its own submission (this slice is the resident-I/O seam, not yet a
 * single fused command list / deferred fence — that is GEMMA-WARP-13b-3, which can layer
 * {@code DirectMlGpuBatch} on top). Stateless and cheap; create one per pipeline run.</p>
 */
public final class WarpExecutionContext {

    private final WindowsBindings wb;

    public WarpExecutionContext(WindowsBindings wb) {
        this.wb = Objects.requireNonNull(wb, "wb");
    }

    public WindowsBindings bindings() {
        return wb;
    }

    /**
     * A resident output buffer of {@code elementCount} floats (GEMMA-WARP-13b-3b: <b>uninitialised</b> —
     * no zero-init upload). Resident kernels fully overwrite their output before it is read, so skipping
     * the per-allocation upload removes one submit + one fence wait per kernel output. See
     * {@link WarpGpuBuffer#allocateUninitialized}.
     */
    public WarpGpuBuffer allocate(int elementCount) throws WindowsNativeException {
        return WarpGpuBuffer.allocateUninitialized(wb, elementCount);
    }

    /** A resident buffer initialised from {@code data}. */
    public WarpGpuBuffer upload(float[] data) throws WindowsNativeException {
        return WarpGpuBuffer.upload(wb, data);
    }

    /**
     * Record + submit one compute-kernel dispatch over GPU-resident buffers (no readback). The
     * {@code uavAddresses} are {@link WarpGpuBuffer#gpuAddress()} values in the kernel's UAV order.
     *
     * <p>When a {@link com.aresstack.windirectml.runtime.DirectMlGpuBatch} is active on the calling
     * thread (GEMMA-WARP-13b-3b) the dispatch is submitted fire-and-forget via
     * {@link D3D12Bindings#executeOrDefer} — its fence is coalesced into the batch drain instead of a
     * per-dispatch CPU wait. Without a batch this behaves exactly as before (one submit + one fence
     * wait). Either way the command list and allocator are released here (the batch holds its own
     * {@code AddRef} until the drain), so this no longer leaks one command list/allocator per dispatch.</p>
     */
    public void dispatch(GpuComputeKernel kernel, long[] uavAddresses, int[] constants, int elementCount)
            throws WindowsNativeException {
        Objects.requireNonNull(kernel, "kernel");
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment queue = wb.getCommandQueue();
        try (Arena a = Arena.ofConfined()) {
            MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                    dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
            MemorySegment cmdList = D3D12Bindings.createCommandList(
                    dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
            kernel.recordDispatch(cmdList, uavAddresses, constants, elementCount);
            D3D12Bindings.executeOrDefer(dev, queue, cmdList, allocator, a);
            DxgiBindings.release(cmdList);
            DxgiBindings.release(allocator);
        }
    }
}
