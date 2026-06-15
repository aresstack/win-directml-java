package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.DxgiBindings;
import com.aresstack.windirectml.windows.GpuComputeKernel;
import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * WARP/DirectML fused GeGLU kernel for Gemma 3 (GEMMA-WARP-6).
 *
 * <p>{@code out[i] = gelu_tanh(gate[i]) * up[i]} over a fused {@code [gate | up]} input of width
 * {@code 2*intermediate}. The GPU mirror of {@link Gemma3ReferenceMath#geluTanh} followed by
 * {@link Gemma3ReferenceMath#multiplyInPlace}. Gemma's GELU-tanh GeGLU, kept distinct from the
 * Qwen/SmolLM2 SiLU SwiGLU. The HLSL is compiled once; each call uploads, dispatches and reads back.
 * Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpGeGluKernel implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpGeGluKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.device = windowsBindings.getD3d12Device();
        this.queue = windowsBindings.getCommandQueue();
        this.arena = Arena.ofShared();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpActivations.GEGLU_HLSL, "gemma3_geglu",
                2, 1, Gemma3WarpActivations.GROUP_SIZE);
    }

    /**
     * Fused GeGLU over a {@code [gate | up]} vector of length {@code 2*intermediate}: returns
     * {@code gelu_tanh(gate) * up} of length {@code intermediate}.
     */
    public float[] apply(float[] gateUp, int intermediate) throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(gateUp, "gateUp");
        if (intermediate < 1) {
            throw new IllegalArgumentException("intermediate must be positive: " + intermediate);
        }
        if (gateUp.length != 2L * intermediate) {
            throw new IllegalArgumentException("gateUp length must equal 2*intermediate: gateUp="
                    + gateUp.length + ", expected=" + (2L * intermediate));
        }
        try (Arena a = Arena.ofConfined()) {
            MemorySegment inBuf = D3D12Bindings.createDefaultBuffer(device, gateUp.length * 4L, a);
            MemorySegment outBuf = D3D12Bindings.createDefaultBuffer(device, intermediate * 4L, a);
            try {
                D3D12Bindings.uploadFloats(device, queue, inBuf, gateUp, a);
                D3D12Bindings.uploadFloats(device, queue, outBuf, new float[intermediate], a);

                MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                MemorySegment cmdList = D3D12Bindings.createCommandList(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(inBuf),
                        D3D12Bindings.getGpuVirtualAddress(outBuf)
                };
                int[] constants = {intermediate};
                kernel.recordDispatch(cmdList, uavs, constants, intermediate);
                D3D12Bindings.executeAndWait(device, queue, cmdList, a);

                return D3D12Bindings.readbackFloats(device, queue, outBuf, intermediate, a);
            } finally {
                DxgiBindings.release(inBuf);
                DxgiBindings.release(outBuf);
            }
        }
    }

    /**
     * GPU-resident fused GeGLU (GEMMA-WARP-13b-2): reads a {@code [2*intermediate]} resident
     * {@code [gate|up]} buffer and writes {@code [intermediate]} to a new resident buffer — no
     * upload/readback. Same math as {@link #apply(float[], int)}.
     */
    public WarpGpuBuffer apply(WarpExecutionContext ctx, WarpGpuBuffer gateUp, int intermediate)
            throws WindowsNativeException {
        ensureOpen();
        if (intermediate < 1) {
            throw new IllegalArgumentException("intermediate must be positive: " + intermediate);
        }
        if (gateUp.elementCount() != 2 * intermediate) {
            throw new IllegalArgumentException("gateUp length must equal 2*intermediate: gateUp="
                    + gateUp.elementCount() + ", expected=" + (2 * intermediate));
        }
        WarpGpuBuffer out = ctx.allocate(intermediate);
        ctx.dispatch(kernel,
                new long[]{gateUp.gpuAddress(), out.gpuAddress()},
                new int[]{intermediate},
                intermediate);
        return out;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpGeGluKernel is closed");
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
