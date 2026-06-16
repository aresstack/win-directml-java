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
 * Appends a new token's {@code k}/{@code v} row into a GPU-resident KV cache buffer (GEMMA-WARP-13c):
 * {@code cache[dstOffset + i] = src[i]} for {@code i < count}. It is a UAV write (not a copy), so under a
 * {@link com.aresstack.windirectml.runtime.DirectMlGpuBatch} the {@code executeOrDefer} UAV barrier orders
 * it before the attention dispatch that reads the cache — fitting the same resident model as the other
 * Gemma kernels, with no per-token CPU readback. Requires {@link WindowsBindings#isSupported()}.
 */
public final class Gemma3WarpKvAppendKernel implements AutoCloseable {

    private static final String KV_APPEND_HLSL = """
            RWByteAddressBuffer Src   : register(u0);
            RWByteAddressBuffer Cache : register(u1);
            cbuffer CB : register(b0) { uint count; uint dstOffset; };

            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint i = dtid.x;
                if (i < count) {
                    Cache.Store((dstOffset + i) * 4, Src.Load(i * 4));
                }
            }
            """;

    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpKvAppendKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.arena = Arena.ofShared();
        MemorySegment device = windowsBindings.getD3d12Device();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                KV_APPEND_HLSL, "gemma3_kv_append", 2, 2, Gemma3WarpActivations.GROUP_SIZE);
    }

    /**
     * Write {@code count} floats from {@code src} into {@code cache} starting at element
     * {@code dstElementOffset} (e.g. {@code pos * kvDim}). No allocation, no readback.
     */
    public void append(WarpExecutionContext ctx, WarpGpuBuffer src, WarpGpuBuffer cache,
                       int count, int dstElementOffset) throws WindowsNativeException {
        ensureOpen();
        if (src.elementCount() < count) {
            throw new IllegalArgumentException("src shorter than count: " + src.elementCount() + " < " + count);
        }
        if (cache.elementCount() < dstElementOffset + count) {
            throw new IllegalArgumentException("cache too small: need " + (dstElementOffset + count)
                    + ", have " + cache.elementCount());
        }
        ctx.dispatch(kernel,
                new long[]{src.gpuAddress(), cache.gpuAddress()},
                new int[]{count, dstElementOffset},
                count);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpKvAppendKernel is closed");
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
