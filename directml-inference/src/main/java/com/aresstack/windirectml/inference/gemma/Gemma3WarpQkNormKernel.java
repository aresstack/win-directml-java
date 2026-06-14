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
 * WARP/DirectML QK-Norm kernel for Gemma 3 (GEMMA-WARP-5a).
 *
 * <p>Gemma applies a zero-centered RMSNorm per attention head over {@code head_dim} (q_norm on the
 * query heads, k_norm on the key/value heads), with a single {@code [head_dim]} weight shared across
 * heads. This is the GPU mirror of the per-head
 * {@link Gemma3ReferenceMath#rmsNormZeroCentered}(values, head*headDim, headDim, weight, eps) calls in
 * the verified CPU reference. {@code head_dim} is taken from the caller (Gemma's 256 is decoupled from
 * {@code hidden/heads}); it is never derived as {@code hidden / heads}.</p>
 *
 * <p>All heads are normalized in one dispatch ({@code numHeads} thread groups, group <i>g</i>
 * normalizing head <i>g</i>). The HLSL is compiled once at construction. Requires a Windows host with
 * a DirectML/D3D12 device ({@link WindowsBindings#isSupported()}).</p>
 */
public final class Gemma3WarpQkNormKernel implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final GpuComputeKernel kernel;
    private boolean closed;

    public Gemma3WarpQkNormKernel(WindowsBindings windowsBindings) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.device = windowsBindings.getD3d12Device();
        this.queue = windowsBindings.getCommandQueue();
        this.arena = Arena.ofShared();
        MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdListForMh = D3D12Bindings.createCommandList(
                device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMh,
                Gemma3WarpNorms.QK_NORM_HLSL, "gemma3_qk_norm",
                3, 3, Gemma3WarpNorms.GROUP_SIZE);
    }

    /**
     * Per-head zero-centered RMSNorm of a {@code [numHeads * headDim]} packed head vector, with one
     * shared {@code [headDim]} weight ({@code q_norm} or {@code k_norm}) and {@code eps}, allocating the
     * output. Head {@code h} lives at offset {@code h * headDim}.
     */
    public float[] normalizeHeads(float[] heads, int numHeads, int headDim, float[] weight, float eps)
            throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(heads, "heads");
        Objects.requireNonNull(weight, "weight");
        if (numHeads < 1 || headDim < 1) {
            throw new IllegalArgumentException("numHeads and headDim must be positive: numHeads="
                    + numHeads + ", headDim=" + headDim);
        }
        if (heads.length != (long) numHeads * headDim) {
            throw new IllegalArgumentException("heads length must equal numHeads*headDim: heads="
                    + heads.length + ", expected=" + ((long) numHeads * headDim));
        }
        if (weight.length != headDim) {
            throw new IllegalArgumentException("weight length must equal headDim: weight="
                    + weight.length + ", headDim=" + headDim);
        }
        int total = numHeads * headDim;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment inBuf = D3D12Bindings.createDefaultBuffer(device, total * 4L, a);
            MemorySegment wBuf = D3D12Bindings.createDefaultBuffer(device, headDim * 4L, a);
            MemorySegment outBuf = D3D12Bindings.createDefaultBuffer(device, total * 4L, a);
            try {
                D3D12Bindings.uploadFloats(device, queue, inBuf, heads, a);
                D3D12Bindings.uploadFloats(device, queue, wBuf, weight, a);
                D3D12Bindings.uploadFloats(device, queue, outBuf, new float[total], a);

                MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
                MemorySegment cmdList = D3D12Bindings.createCommandList(
                        device, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
                long[] uavs = {
                        D3D12Bindings.getGpuVirtualAddress(inBuf),
                        D3D12Bindings.getGpuVirtualAddress(wBuf),
                        D3D12Bindings.getGpuVirtualAddress(outBuf)
                };
                int[] constants = {numHeads, headDim, Float.floatToRawIntBits(eps)};
                // numHeads groups: group g normalizes head g over head_dim.
                kernel.recordDispatch(cmdList, uavs, constants, numHeads * Gemma3WarpNorms.GROUP_SIZE);
                D3D12Bindings.executeAndWait(device, queue, cmdList, a);

                return D3D12Bindings.readbackFloats(device, queue, outBuf, total, a);
            } finally {
                DxgiBindings.release(inBuf);
                DxgiBindings.release(wBuf);
                DxgiBindings.release(outBuf);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpQkNormKernel is closed");
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
