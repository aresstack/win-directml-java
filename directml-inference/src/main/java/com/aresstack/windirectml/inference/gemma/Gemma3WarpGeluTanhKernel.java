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
 * WARP/DirectML GELU-tanh activation kernel for Gemma 3 (GEMMA-WARP-6).
 *
 * <p>Element-wise {@code 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))} — the GPU mirror of
 * {@link Gemma3ReferenceMath#geluTanh}. This is the gelu_pytorch_tanh activation Gemma uses, kept
 * separate from the Qwen/SmolLM2 SiLU. The HLSL is compiled once; each call uploads, dispatches and
 * reads back (validation building block). Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpGeluTanhKernel implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpGeluTanhKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.device = windowsBindings.getD3d12Device();
        this.queue = windowsBindings.getCommandQueue();
        this.arena = Arena.ofShared();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpActivations.GELU_TANH_HLSL, "gemma3_gelu_tanh",
                2, 1, Gemma3WarpActivations.GROUP_SIZE);
    }

    /** Apply GELU-tanh element-wise to {@code x}, allocating the output. */
    public float[] apply(float[] x) throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(x, "x");
        int count = x.length;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment inBuf = D3D12Bindings.createDefaultBuffer(device, count * 4L, a);
            MemorySegment outBuf = D3D12Bindings.createDefaultBuffer(device, count * 4L, a);
            try {
                D3D12Bindings.uploadFloats(device, queue, inBuf, x, a);
                D3D12Bindings.uploadFloats(device, queue, outBuf, new float[count], a);

                MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                MemorySegment cmdList = D3D12Bindings.createCommandList(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(inBuf),
                        D3D12Bindings.getGpuVirtualAddress(outBuf)
                };
                int[] constants = {count};
                kernel.recordDispatch(cmdList, uavs, constants, count);
                D3D12Bindings.executeAndWait(device, queue, cmdList, a);

                return D3D12Bindings.readbackFloats(device, queue, outBuf, count, a);
            } finally {
                DxgiBindings.release(inBuf);
                DxgiBindings.release(outBuf);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpGeluTanhKernel is closed");
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
