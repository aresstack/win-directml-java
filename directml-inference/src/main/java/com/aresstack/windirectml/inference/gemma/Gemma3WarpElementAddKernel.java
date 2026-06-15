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
 * WARP element-wise add for Gemma residual connections (GEMMA-WARP-13b-3a): {@code out = a + b}. Used by
 * the resident decode step so the two residual adds stay on the GPU (no readback). Requires
 * {@link WindowsBindings#isSupported()}.
 */
public final class Gemma3WarpElementAddKernel implements AutoCloseable {

    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpElementAddKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.arena = Arena.ofShared();
        MemorySegment device = windowsBindings.getD3d12Device();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpActivations.ELEMENT_ADD_HLSL, "gemma3_element_add",
                3, 1, Gemma3WarpActivations.GROUP_SIZE);
    }

    /** GPU-resident {@code out = a + b} (equal-length buffers), allocating the resident output. */
    public WarpGpuBuffer add(WarpExecutionContext ctx, WarpGpuBuffer a, WarpGpuBuffer b)
            throws WindowsNativeException {
        ensureOpen();
        int n = a.elementCount();
        if (b.elementCount() != n) {
            throw new IllegalArgumentException("a and b must have equal length: a=" + n + ", b=" + b.elementCount());
        }
        WarpGpuBuffer out = ctx.allocate(n);
        ctx.dispatch(kernel,
                new long[]{a.gpuAddress(), b.gpuAddress(), out.gpuAddress()},
                new int[]{n},
                n);
        return out;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpElementAddKernel is closed");
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
