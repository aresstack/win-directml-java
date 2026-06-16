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
 * WARP/DirectML fused zero-centered RMSNorm + residual add for Gemma 3 (GEMMA-WARP-15).
 *
 * <p>Computes {@code out = residual + rmsNorm(x, weight)} in a single dispatch — the GPU mirror of running
 * {@link Gemma3WarpRmsNormKernel} then {@link Gemma3WarpElementAddKernel}. Gemma applies a post-norm to
 * each sublayer output before adding it to the residual (post-attention and post-feedforward), so the
 * resident decode/prefill path used two dispatches (norm, then add) with a UAV barrier between. Folding the
 * add into the norm's final store loop removes one dispatch + one UAV barrier per sublayer (2 per layer)
 * with no extra serial work. Byte-identical to the two-kernel path: same float operations, same order
 * ({@code r + v*rms_inv*(1+w)}).</p>
 *
 * <p>Single-group reduction like {@link Gemma3WarpRmsNormKernel} (one D3D12 thread group strides over the
 * whole vector), so it serves the hidden-state norm (dim=640). Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpFusedNormAddKernel implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpFusedNormAddKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.device = windowsBindings.getD3d12Device();
        this.queue = windowsBindings.getCommandQueue();
        this.arena = Arena.ofShared();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpNorms.ZERO_CENTERED_RMSNORM_ADD_HLSL, "gemma3_zero_centered_rmsnorm_add",
                4, 2, Gemma3WarpNorms.GROUP_SIZE);
    }

    /**
     * Fused {@code out = residual + rmsNorm(x, weight)} over {@code dim} values, allocating the output
     * (validation building block: upload/dispatch/readback). Mirrors {@link #normAdd(WarpExecutionContext,
     * WarpGpuBuffer, WarpGpuBuffer, WarpGpuBuffer, float)}.
     */
    public float[] normAdd(float[] x, float[] weight, float[] residual, float eps) throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(residual, "residual");
        int dim = x.length;
        if (weight.length != dim || residual.length != dim) {
            throw new IllegalArgumentException("x, weight and residual must have equal length: x=" + dim
                    + ", weight=" + weight.length + ", residual=" + residual.length);
        }
        try (Arena a = Arena.ofConfined()) {
            MemorySegment inBuf = D3D12Bindings.createDefaultBuffer(device, dim * 4L, a);
            MemorySegment wBuf = D3D12Bindings.createDefaultBuffer(device, dim * 4L, a);
            MemorySegment rBuf = D3D12Bindings.createDefaultBuffer(device, dim * 4L, a);
            MemorySegment outBuf = D3D12Bindings.createDefaultBuffer(device, dim * 4L, a);
            try {
                D3D12Bindings.uploadFloats(device, queue, inBuf, x, a);
                D3D12Bindings.uploadFloats(device, queue, wBuf, weight, a);
                D3D12Bindings.uploadFloats(device, queue, rBuf, residual, a);
                D3D12Bindings.uploadFloats(device, queue, outBuf, new float[dim], a);

                MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                MemorySegment cmdList = D3D12Bindings.createCommandList(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(inBuf),
                        D3D12Bindings.getGpuVirtualAddress(wBuf),
                        D3D12Bindings.getGpuVirtualAddress(rBuf),
                        D3D12Bindings.getGpuVirtualAddress(outBuf)
                };
                int[] constants = {dim, Float.floatToRawIntBits(eps)};
                kernel.recordDispatch(cmdList, uavs, constants, Gemma3WarpNorms.GROUP_SIZE);
                D3D12Bindings.executeAndWait(device, queue, cmdList, a);

                return D3D12Bindings.readbackFloats(device, queue, outBuf, dim, a);
            } finally {
                DxgiBindings.release(inBuf);
                DxgiBindings.release(wBuf);
                DxgiBindings.release(rBuf);
                DxgiBindings.release(outBuf);
            }
        }
    }

    /**
     * GPU-resident fused {@code out = residual + rmsNorm(x, weight)} (GEMMA-WARP-15): reads {@code x},
     * {@code weight} and {@code residual} from resident buffers and writes a new resident buffer — no
     * upload/readback. One dispatch instead of the RMSNorm + element-add pair.
     */
    public WarpGpuBuffer normAdd(WarpExecutionContext ctx, WarpGpuBuffer x, WarpGpuBuffer weight,
                                 WarpGpuBuffer residual, float eps) throws WindowsNativeException {
        ensureOpen();
        int dim = x.elementCount();
        if (weight.elementCount() != dim || residual.elementCount() != dim) {
            throw new IllegalArgumentException("x, weight and residual must have equal length: x=" + dim
                    + ", weight=" + weight.elementCount() + ", residual=" + residual.elementCount());
        }
        WarpGpuBuffer out = ctx.allocate(dim);
        ctx.dispatch(kernel,
                new long[]{x.gpuAddress(), weight.gpuAddress(), residual.gpuAddress(), out.gpuAddress()},
                new int[]{dim, Float.floatToRawIntBits(eps)},
                Gemma3WarpNorms.GROUP_SIZE);
        return out;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpFusedNormAddKernel is closed");
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
