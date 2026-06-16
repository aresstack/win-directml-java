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
 * Copies a contiguous run of floats between resident buffers at arbitrary element offsets
 * (GEMMA-WARP-13e): {@code dst[dstOffset + i] = src[srcOffset + i]} for {@code i < count}. A UAV write, so
 * under a coalescing recording it shares the command list with the surrounding kernels (no extra submit).
 *
 * <p>Used by batched prefill to <b>gather</b> per-position vectors into a contiguous projection input
 * ({@code dstOffset = pos*dim, srcOffset = 0}) and to <b>scatter</b> a projection's contiguous batch output
 * back into per-position buffers ({@code dstOffset = 0, srcOffset = pos*dim}). Requires
 * {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpRowCopyKernel implements AutoCloseable {

    private static final String ROW_COPY_HLSL = """
            RWByteAddressBuffer Src : register(u0);
            RWByteAddressBuffer Dst : register(u1);
            cbuffer CB : register(b0) { uint count; uint srcOffset; uint dstOffset; };

            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint i = dtid.x;
                if (i < count) {
                    Dst.Store((dstOffset + i) * 4, Src.Load((srcOffset + i) * 4));
                }
            }
            """;

    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpRowCopyKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.arena = Arena.ofShared();
        MemorySegment device = windowsBindings.getD3d12Device();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                ROW_COPY_HLSL, "gemma3_row_copy", 2, 3, Gemma3WarpActivations.GROUP_SIZE);
    }

    /** Copy {@code count} floats {@code dst[dstOffset + i] = src[srcOffset + i]} (resident, no readback). */
    public void copy(WarpExecutionContext ctx, WarpGpuBuffer src, int srcElementOffset,
                     WarpGpuBuffer dst, int dstElementOffset, int count) throws WindowsNativeException {
        ensureOpen();
        if (src.elementCount() < srcElementOffset + count) {
            throw new IllegalArgumentException("src too small: need " + (srcElementOffset + count)
                    + ", have " + src.elementCount());
        }
        if (dst.elementCount() < dstElementOffset + count) {
            throw new IllegalArgumentException("dst too small: need " + (dstElementOffset + count)
                    + ", have " + dst.elementCount());
        }
        ctx.dispatch(kernel,
                new long[]{src.gpuAddress(), dst.gpuAddress()},
                new int[]{count, srcElementOffset, dstElementOffset},
                count);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpRowCopyKernel is closed");
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
