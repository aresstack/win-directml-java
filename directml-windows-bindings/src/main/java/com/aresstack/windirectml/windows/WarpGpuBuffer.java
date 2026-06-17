package com.aresstack.windirectml.windows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * A device-resident FP32 GPU buffer for chaining WARP/DirectML compute kernels without per-step CPU
 * round-trips (GEMMA-WARP-13b-2).
 *
 * <p>Holds one D3D12 default buffer of {@code elementCount} floats, usable directly as a
 * {@code RWByteAddressBuffer} UAV (same usage pattern the per-call Gemma kernels already rely on).
 * {@link #upload}/{@link #allocate} create it (one submit), {@link #gpuAddress()} feeds a kernel
 * dispatch, and {@link #readback()} pulls the result back to the CPU — typically only the final result,
 * so intermediate tensors stay GPU-resident. Each buffer owns its allocation arena; {@link #close()}
 * releases it.</p>
 */
public final class WarpGpuBuffer implements AutoCloseable {

    private final MemorySegment device;
    private final MemorySegment queue;
    private final Arena arena;
    private final MemorySegment buffer;
    private final int elementCount;
    // GEMMA-WARP-16: a slice view shares the parent's D3D12 buffer at a byte offset and does not own the
    // allocation (arena == null). Used to read a sub-region (e.g. the Q/K/V parts of a fused QKV matmul
    // output) as a kernel UAV without a copy. byteOffset is 0 for an owning buffer.
    private final long byteOffset;
    private final boolean view;
    private boolean closed;

    private WarpGpuBuffer(MemorySegment device, MemorySegment queue, Arena arena,
                          MemorySegment buffer, int elementCount) {
        this(device, queue, arena, buffer, elementCount, 0L, false);
    }

    private WarpGpuBuffer(MemorySegment device, MemorySegment queue, Arena arena,
                          MemorySegment buffer, int elementCount, long byteOffset, boolean view) {
        this.device = device;
        this.queue = queue;
        this.arena = arena;
        this.buffer = buffer;
        this.elementCount = elementCount;
        this.byteOffset = byteOffset;
        this.view = view;
    }

    /** Allocate a zero-initialised resident buffer of {@code elementCount} floats. */
    public static WarpGpuBuffer allocate(WindowsBindings wb, int elementCount) throws WindowsNativeException {
        if (elementCount < 1) {
            throw new IllegalArgumentException("elementCount must be positive: " + elementCount);
        }
        Arena a = Arena.ofShared();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();
        MemorySegment buf = D3D12Bindings.createDefaultBuffer(dev, (long) elementCount * Float.BYTES, a);
        // Upload zeros once to initialise + place the buffer in the UAV-usable state the kernels expect.
        D3D12Bindings.uploadFloats(dev, q, buf, new float[elementCount], a);
        return new WarpGpuBuffer(dev, q, a, buf, elementCount);
    }

    /**
     * Allocate a resident buffer of {@code elementCount} floats <b>without</b> the zero-init upload
     * (GEMMA-WARP-13b-3b). The contents are undefined; this is for kernel <i>output</i> buffers that the
     * dispatch fully overwrites before any read. Saving the per-allocation upload removes one submit +
     * one fence wait per resident kernel output — the dominant remaining sync cost of the resident decode
     * — without changing results. A D3D12 default-heap buffer is created in COMMON state and implicitly
     * promotes to UNORDERED_ACCESS / COPY_DEST on first GPU use, so no explicit init is required.
     *
     * <p>Only safe when the buffer is fully written before being read. All resident Gemma kernels
     * (RMSNorm, QK-norm, RoPE, attention scores incl. the masked {@code -1e30f} sentinel, softmax,
     * value, GeGLU, element-add and the {@code matvecResident} projections) write every output element,
     * so {@link WarpExecutionContext#allocate} uses this; the zero-initialising {@link #allocate} stays
     * the default for any caller that needs defined contents.</p>
     */
    public static WarpGpuBuffer allocateUninitialized(WindowsBindings wb, int elementCount) throws WindowsNativeException {
        if (elementCount < 1) {
            throw new IllegalArgumentException("elementCount must be positive: " + elementCount);
        }
        Arena a = Arena.ofShared();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();
        MemorySegment buf = D3D12Bindings.createDefaultBuffer(dev, (long) elementCount * Float.BYTES, a);
        return new WarpGpuBuffer(dev, q, a, buf, elementCount);
    }

    /** Create a resident buffer initialised from {@code data} (one upload). */
    public static WarpGpuBuffer upload(WindowsBindings wb, float[] data) throws WindowsNativeException {
        if (data.length < 1) {
            throw new IllegalArgumentException("data must not be empty");
        }
        Arena a = Arena.ofShared();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();
        MemorySegment buf = D3D12Bindings.createDefaultBuffer(dev, (long) data.length * Float.BYTES, a);
        D3D12Bindings.uploadFloats(dev, q, buf, data, a);
        return new WarpGpuBuffer(dev, q, a, buf, data.length);
    }

    public int elementCount() {
        return elementCount;
    }

    /** The buffer's GPU virtual address (plus the slice offset for a view), for use as a kernel UAV. */
    public long gpuAddress() {
        ensureOpen();
        return D3D12Bindings.getGpuVirtualAddress(buffer) + byteOffset;
    }

    /**
     * A non-owning slice view of {@code sliceCount} floats starting at element {@code elementOffset}
     * (GEMMA-WARP-16). It shares this buffer's D3D12 resource at the corresponding byte offset and is
     * usable as a kernel UAV input ({@link #gpuAddress()} returns the offset address; the 4-byte element
     * offset satisfies the D3D12 root-descriptor alignment). The view does not own the allocation:
     * {@link #close()} is a no-op and it must not outlive the parent buffer. It cannot be sliced again,
     * read back, or used as a resident-matvec output target.
     */
    public WarpGpuBuffer slice(int elementOffset, int sliceCount) {
        ensureOpen();
        if (view) {
            throw new IllegalStateException("cannot slice a slice view");
        }
        if (elementOffset < 0 || sliceCount < 1 || (long) elementOffset + sliceCount > elementCount) {
            throw new IllegalArgumentException("slice [" + elementOffset + ", " + (elementOffset + sliceCount)
                    + ") out of bounds for elementCount=" + elementCount);
        }
        return new WarpGpuBuffer(device, queue, null, buffer, sliceCount,
                byteOffset + (long) elementOffset * Float.BYTES, true);
    }

    /** The underlying D3D12 buffer resource (package-internal: for GPU→GPU copies, e.g. resident matvec). */
    MemorySegment d3d12Buffer() {
        ensureOpen();
        if (view) {
            throw new IllegalStateException("a slice view has no standalone D3D12 buffer (offset binding only)");
        }
        return buffer;
    }

    /** Copy the buffer contents back to a host {@code float[]} (one readback). */
    public float[] readback() throws WindowsNativeException {
        ensureOpen();
        if (view) {
            throw new IllegalStateException("cannot read back a slice view");
        }
        try (Arena a = Arena.ofConfined()) {
            return D3D12Bindings.readbackFloats(device, queue, buffer, elementCount, a);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("WarpGpuBuffer is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (!view) {
                // A slice view does not own the D3D12 buffer/arena — only the parent releases them.
                DxgiBindings.release(buffer);
                arena.close();
            }
        }
    }
}
