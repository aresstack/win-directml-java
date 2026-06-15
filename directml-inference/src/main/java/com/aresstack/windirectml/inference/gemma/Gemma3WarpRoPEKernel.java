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
 * WARP/DirectML rotate-half RoPE kernel for Gemma 3 (GEMMA-WARP-7b).
 *
 * <p>The GPU mirror of {@link Gemma3RoPE#applyToHeads} / {@link Gemma3ReferenceMath#applyRopeHalf}:
 * GPT-NeoX rotate-half over each {@code head_dim}-wide head of a packed {@code [numHeads * head_dim]}
 * q/k vector, for one sequence position, with a per-layer base theta (dual local/global). The HLSL is
 * compiled once; each call uploads, dispatches and reads back. Requires
 * {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpRoPEKernel implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpRoPEKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.device = windowsBindings.getD3d12Device();
        this.queue = windowsBindings.getCommandQueue();
        this.arena = Arena.ofShared();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpAttention.ROPE_HLSL, "gemma3_rope",
                2, 4, Gemma3WarpAttention.GROUP_SIZE);
    }

    /**
     * Apply rotate-half RoPE to a packed {@code [numHeads * headDim]} vector at position {@code pos}
     * with base {@code theta}, allocating the output. {@code headDim} must be even.
     */
    public float[] applyToHeads(float[] packed, int numHeads, int headDim, int pos, float theta)
            throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(packed, "packed");
        if (numHeads < 1 || headDim < 1) {
            throw new IllegalArgumentException("numHeads and headDim must be positive: numHeads="
                    + numHeads + ", headDim=" + headDim);
        }
        if ((headDim & 1) != 0) {
            throw new IllegalArgumentException("headDim must be even: " + headDim);
        }
        if (packed.length != (long) numHeads * headDim) {
            throw new IllegalArgumentException("packed length must equal numHeads*headDim: packed="
                    + packed.length + ", expected=" + ((long) numHeads * headDim));
        }
        int total = numHeads * headDim;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment inBuf = D3D12Bindings.createDefaultBuffer(device, total * 4L, a);
            MemorySegment outBuf = D3D12Bindings.createDefaultBuffer(device, total * 4L, a);
            try {
                D3D12Bindings.uploadFloats(device, queue, inBuf, packed, a);
                D3D12Bindings.uploadFloats(device, queue, outBuf, new float[total], a);

                MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                MemorySegment cmdList = D3D12Bindings.createCommandList(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(inBuf),
                        D3D12Bindings.getGpuVirtualAddress(outBuf)
                };
                int[] constants = {numHeads, headDim, pos, Float.floatToRawIntBits(theta)};
                int threads = numHeads * (headDim / 2); // one thread per (head, i<half)
                kernel.recordDispatch(cmdList, uavs, constants, threads);
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
            throw new IllegalStateException("Gemma3WarpRoPEKernel is closed");
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
