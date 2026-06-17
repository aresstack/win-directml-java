package com.aresstack.windirectml.windows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Minimal GPU-resident execution seam for chaining WARP/DirectML compute kernels (GEMMA-WARP-13b-2).
 *
 * <p>Provides resident {@link WarpGpuBuffer} factories and a {@link #dispatch} that records one compute
 * kernel into a command list and submits it (one submit + fence wait, counted by
 * {@link WarpSubmissionStats}) <b>without any CPU readback</b> — the output stays in a GPU buffer for the
 * next kernel. This removes the per-kernel upload→dispatch→readback→re-upload round-trips of the
 * float[] API; the only readback is the caller's final {@link WarpGpuBuffer#readback()}.</p>
 *
 * <p>Each {@code dispatch} is still its own submission (this slice is the resident-I/O seam, not yet a
 * single fused command list / deferred fence — that is GEMMA-WARP-13b-3, which can layer
 * {@code DirectMlGpuBatch} on top). Stateless and cheap; create one per pipeline run.</p>
 */
public final class WarpExecutionContext {

    private final WindowsBindings wb;

    // GEMMA-WARP-13d/13g command-list coalescing: inside a coalescing scope (beginRecording..flushRecording),
    // the layer's dispatches AND its resident matvec projections accumulate into ONE command list (a UAV
    // barrier after each) and submit once. The matvec records via MatMulNBitsKernel.recordResidentInto
    // (its dispatch + staging copies unchanged, in/out bracketed with UAV<->COPY transitions), so the math
    // is byte-identical to the non-coalesced path. The list is opened lazily on the first dispatch/matvec.
    // The single batched-prefill matmul (matvecBatched) still flushes + runs synchronously (shared scratch).
    private boolean coalescing;
    private Arena recArena;
    private MemorySegment recAllocator;
    private MemorySegment recCmdList;
    private boolean recording; // a command list is currently open within the coalescing scope

    // GEMMA-WARP-17: opt-in coarse group timing. When a sink is attached the caller still coalesces each
    // group's dispatches into one command list (as the product path does) but runs WITHOUT a deferred
    // batch, so mark()/endGroup() flush+fence the group's list synchronously and attribute the elapsed
    // wall-clock + recorded-dispatch delta to that group — one submit/fence per group, representative of
    // the coalesced pipeline. Null in the normal runtime (mark()/endGroup() return immediately, zero cost).
    private WarpGroupSink groupSink;
    private String curGroup;
    private long curGroupT0;
    private long curGroupDispatchBefore;

    public WarpExecutionContext(WindowsBindings wb) {
        this.wb = Objects.requireNonNull(wb, "wb");
    }

    /** Attach (or clear with {@code null}) the opt-in per-group timing sink (GEMMA-WARP-17). */
    public void setGroupSink(WarpGroupSink sink) {
        this.groupSink = sink;
    }

    /** Whether group timing is active — callers skip coalescing/the deferred batch so groups are measurable. */
    public boolean isGroupProfiling() {
        return groupSink != null;
    }

    /**
     * Close the current timing group (flushing+fencing its coalesced command list) and start a new one named
     * {@code name} (GEMMA-WARP-17). No-op when no sink is attached.
     */
    public void mark(String name) throws WindowsNativeException {
        if (groupSink == null) {
            return;
        }
        closeCurrentGroup();
        curGroup = name;
        curGroupT0 = System.nanoTime();
        curGroupDispatchBefore = WarpSubmissionStats.snapshot().dispatches();
    }

    /** Close the current timing group without starting a new one (e.g. end of the tail). No-op when off. */
    public void endGroup() throws WindowsNativeException {
        if (groupSink == null) {
            return;
        }
        closeCurrentGroup();
    }

    private void closeCurrentGroup() throws WindowsNativeException {
        if (curGroup == null) {
            return;
        }
        // Flush+fence the group's coalesced list (no batch active in profile mode → executeAndWait), so the
        // elapsed wall-clock is the group's GPU time for one submit/fence.
        flushOpenList();
        long nanos = System.nanoTime() - curGroupT0;
        long dispatches = WarpSubmissionStats.snapshot().dispatches() - curGroupDispatchBefore;
        groupSink.record(curGroup, dispatches, nanos);
        curGroup = null;
    }

    public WindowsBindings bindings() {
        return wb;
    }

    /**
     * Open a command-list coalescing scope (GEMMA-WARP-13d): within it, consecutive {@link #dispatch} (UAV)
     * kernels accumulate into one command list submitted once, instead of one submission per kernel.
     * {@link #matvec} flushes the pending list and runs standalone. Use only for a sequence with no
     * intervening readback/upload (those need the GPU drained); the resident decode layer qualifies. Not
     * re-entrant. Pair with {@link #flushRecording}.
     */
    public void beginRecording() {
        if (coalescing) {
            throw new IllegalStateException("coalescing scope already open");
        }
        coalescing = true;
    }

    /** Whether a coalescing scope is open. */
    public boolean isRecording() {
        return coalescing;
    }

    /** Close the coalescing scope, submitting any pending command list (synchronous when no batch). */
    public void flushRecording() throws WindowsNativeException {
        flushOpenList();
        coalescing = false;
    }

    /**
     * Run a resident matvec {@code out = W·in}. Inside a coalescing scope (GEMMA-WARP-13g) it records the
     * matvec into the shared layer command list via {@link MatMulNBitsKernel#recordResidentInto} (the
     * projection shares the layer's single submit); the matvec dispatch + staging copies are unchanged, so
     * the math is byte-identical. Outside a scope it runs the standalone resident matvec. Used by
     * {@code WarpDenseProjection.forwardResident}.
     */
    public void matvec(MatMulNBitsKernel kernel, WarpGpuBuffer in, WarpGpuBuffer out)
            throws WindowsNativeException {
        Objects.requireNonNull(kernel, "kernel");
        if (coalescing) {
            // GEMMA-WARP-13g: record the matvec INTO the shared layer command list (no flush, no separate
            // submit) — the projection joins the layer's single command list. Numerics unchanged.
            openListIfNeeded();
            kernel.recordResidentInto(recCmdList, in, out, recArena);
            D3D12Bindings.uavBarrier(recCmdList, recArena);
            return;
        }
        kernel.matvecResident(in, out);
    }

    /**
     * Run a resident <b>batched</b> matmul {@code out[rows,N] = in[rows,K] · Wᵀ} (GEMMA-WARP-13e). Inside a
     * coalescing scope it first flushes the pending UAV list (so the gathered input is on the GPU), then
     * runs the kernel's synchronous resident batched matmul (the shared batch scratch can't be deferred).
     * Used by {@code WarpDenseProjection.forwardResidentBatched} for batched prefill projections.
     */
    public void matvecBatched(MatMulNBitsKernel kernel, WarpGpuBuffer in, WarpGpuBuffer out, int rows)
            throws WindowsNativeException {
        Objects.requireNonNull(kernel, "kernel");
        if (coalescing) {
            flushOpenList();
        }
        kernel.matmulBatchResident(in, out, rows);
    }

    private void openListIfNeeded() throws WindowsNativeException {
        if (recording) {
            return;
        }
        MemorySegment dev = wb.getD3d12Device();
        recArena = Arena.ofConfined();
        recAllocator = D3D12Bindings.createCommandAllocator(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, recArena);
        recCmdList = D3D12Bindings.createCommandList(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, recAllocator, recArena);
        recording = true;
    }

    private void flushOpenList() throws WindowsNativeException {
        if (!recording) {
            return;
        }
        recording = false;
        try {
            D3D12Bindings.executeOrDefer(wb.getD3d12Device(), wb.getCommandQueue(), recCmdList, recAllocator, recArena);
            DxgiBindings.release(recCmdList);
            DxgiBindings.release(recAllocator);
        } finally {
            recArena.close();
            recArena = null;
            recCmdList = null;
            recAllocator = null;
        }
    }

    /**
     * A resident output buffer of {@code elementCount} floats (GEMMA-WARP-13b-3b: <b>uninitialised</b> —
     * no zero-init upload). Resident kernels fully overwrite their output before it is read, so skipping
     * the per-allocation upload removes one submit + one fence wait per kernel output. See
     * {@link WarpGpuBuffer#allocateUninitialized}.
     */
    public WarpGpuBuffer allocate(int elementCount) throws WindowsNativeException {
        return WarpGpuBuffer.allocateUninitialized(wb, elementCount);
    }

    /** A resident buffer initialised from {@code data}. */
    public WarpGpuBuffer upload(float[] data) throws WindowsNativeException {
        return WarpGpuBuffer.upload(wb, data);
    }

    /**
     * GPU→GPU copy of {@code elementCount} floats from {@code src[srcElementOffset]} into
     * {@code dst[dstElementOffset]} (GEMMA-WARP-13c), framed by COMMON↔COPY_DEST transition barriers so the
     * copy is ordered against later use of {@code dst}. Intended for infrequent resident KV-cache growth
     * <b>outside</b> a {@link com.aresstack.windirectml.runtime.DirectMlGpuBatch} (then it is synchronous,
     * so the source/old buffer may be freed right after). Per-token cache appends use the UAV-write append
     * kernel instead, which fits the deferred batch's UAV-barrier model.
     */
    public void copyRegionInto(WarpGpuBuffer dst, int dstElementOffset, WarpGpuBuffer src,
                               int srcElementOffset, int elementCount) throws WindowsNativeException {
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment queue = wb.getCommandQueue();
        long bytes = (long) elementCount * Float.BYTES;
        long dstOff = (long) dstElementOffset * Float.BYTES;
        long srcOff = (long) srcElementOffset * Float.BYTES;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                    dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
            MemorySegment cmdList = D3D12Bindings.createCommandList(
                    dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
            D3D12Bindings.transitionBarrier(cmdList, dst.d3d12Buffer(),
                    D3D12Bindings.D3D12_RESOURCE_STATE_COMMON, D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST, a);
            D3D12Bindings.copyBufferRegion(cmdList, dst.d3d12Buffer(), dstOff, src.d3d12Buffer(), srcOff, bytes);
            D3D12Bindings.transitionBarrier(cmdList, dst.d3d12Buffer(),
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST, D3D12Bindings.D3D12_RESOURCE_STATE_COMMON, a);
            D3D12Bindings.executeOrDefer(dev, queue, cmdList, allocator, a);
            DxgiBindings.release(cmdList);
            DxgiBindings.release(allocator);
        }
    }

    /**
     * Record + submit one compute-kernel dispatch over GPU-resident buffers (no readback). The
     * {@code uavAddresses} are {@link WarpGpuBuffer#gpuAddress()} values in the kernel's UAV order.
     *
     * <p>When a {@link com.aresstack.windirectml.runtime.DirectMlGpuBatch} is active on the calling
     * thread (GEMMA-WARP-13b-3b) the dispatch is submitted fire-and-forget via
     * {@link D3D12Bindings#executeOrDefer} — its fence is coalesced into the batch drain instead of a
     * per-dispatch CPU wait. Without a batch this behaves exactly as before (one submit + one fence
     * wait). Either way the command list and allocator are released here (the batch holds its own
     * {@code AddRef} until the drain), so this no longer leaks one command list/allocator per dispatch.</p>
     */
    public void dispatch(GpuComputeKernel kernel, long[] uavAddresses, int[] constants, int elementCount)
            throws WindowsNativeException {
        Objects.requireNonNull(kernel, "kernel");
        if (coalescing) {
            // GEMMA-WARP-13d: accumulate into one command list (lazily opened) + a UAV barrier so the next
            // dependent dispatch observes this one's writes; submitted once by the next matvec flush or by
            // flushRecording().
            openListIfNeeded();
            kernel.recordDispatch(recCmdList, uavAddresses, constants, elementCount);
            D3D12Bindings.uavBarrier(recCmdList, recArena);
            return;
        }
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment queue = wb.getCommandQueue();
        try (Arena a = Arena.ofConfined()) {
            MemorySegment allocator = D3D12Bindings.createCommandAllocator(
                    dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, a);
            MemorySegment cmdList = D3D12Bindings.createCommandList(
                    dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, a);
            kernel.recordDispatch(cmdList, uavAddresses, constants, elementCount);
            D3D12Bindings.executeOrDefer(dev, queue, cmdList, allocator, a);
            DxgiBindings.release(cmdList);
            DxgiBindings.release(allocator);
        }
    }
}
