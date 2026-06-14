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
 * WARP/DirectML zero-centered RMSNorm kernel for Gemma 3 (GEMMA-WARP-5a).
 *
 * <p>The GPU mirror of {@link Gemma3ReferenceMath#rmsNormZeroCentered}: {@code y = x * rsqrt(mean(x^2)
 * + eps) * (1 + weight)}. Distinct from the Phi-3/Qwen RMSNorm (which scales by {@code weight}, not
 * {@code 1 + weight}) — see {@link Gemma3WarpNorms}. The whole vector is reduced by a single D3D12
 * thread group, so it serves both the hidden-state norms (dim=640) and, conceptually, a per-head norm
 * (dim=head_dim) — though per-head QK-norm is its own kernel
 * ({@link Gemma3WarpQkNormKernel}) so all heads run in one dispatch.</p>
 *
 * <p>The HLSL is compiled once at construction; each {@link #normalize} call uploads its inputs, runs
 * one dispatch and reads the result back (validation building block, not yet the fused product path).
 * Requires a Windows host with a DirectML/D3D12 device ({@link WindowsBindings#isSupported()}).</p>
 */
public final class Gemma3WarpRmsNormKernel implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    /** Compile the kernel against an initialised {@link WindowsBindings} (device already up). */
    public Gemma3WarpRmsNormKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.device = windowsBindings.getD3d12Device();
        this.queue = windowsBindings.getCommandQueue();
        this.arena = Arena.ofShared();
        // A throwaway command list only to bind the vtable method handles; dispatches use fresh lists.
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpNorms.ZERO_CENTERED_RMSNORM_HLSL, "gemma3_zero_centered_rmsnorm",
                3, 2, Gemma3WarpNorms.GROUP_SIZE);
    }

    /**
     * Zero-centered RMSNorm of {@code x} ({@code dim} values) with {@code weight} ({@code dim} values)
     * and {@code eps}, allocating the output. The whole vector is reduced by one thread group.
     */
    public float[] normalize(float[] x, float[] weight, float eps) throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(weight, "weight");
        if (x.length != weight.length) {
            throw new IllegalArgumentException("x and weight must have equal length: x=" + x.length
                    + ", weight=" + weight.length);
        }
        int dim = x.length;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment inBuf = D3D12Bindings.createDefaultBuffer(device, dim * 4L, a);
            MemorySegment wBuf = D3D12Bindings.createDefaultBuffer(device, dim * 4L, a);
            MemorySegment outBuf = D3D12Bindings.createDefaultBuffer(device, dim * 4L, a);
            try {
                D3D12Bindings.uploadFloats(device, queue, inBuf, x, a);
                D3D12Bindings.uploadFloats(device, queue, wBuf, weight, a);
                D3D12Bindings.uploadFloats(device, queue, outBuf, new float[dim], a);

                MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                MemorySegment cmdList = D3D12Bindings.createCommandList(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(inBuf),
                        D3D12Bindings.getGpuVirtualAddress(wBuf),
                        D3D12Bindings.getGpuVirtualAddress(outBuf)
                };
                int[] constants = {dim, Float.floatToRawIntBits(eps)};
                // One group covers the whole vector (the shader strides over dim).
                kernel.recordDispatch(cmdList, uavs, constants, Gemma3WarpNorms.GROUP_SIZE);
                D3D12Bindings.executeAndWait(device, queue, cmdList, a);

                return D3D12Bindings.readbackFloats(device, queue, outBuf, dim, a);
            } finally {
                DxgiBindings.release(inBuf);
                DxgiBindings.release(wBuf);
                DxgiBindings.release(outBuf);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpRmsNormKernel is closed");
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
