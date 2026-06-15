package com.aresstack.windirectml.runtime;

import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.DxgiBindings;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * Thread-bound submission-coalescing helper for DirectML kernel pipelines.
 * <p>
 * Without a batch active, every kernel dispatch creates its own command
 * list, submits it via {@code ID3D12CommandQueue::ExecuteCommandLists},
 * creates a fresh fence, signals + busy-waits, and only then releases the
 * command list / allocator. For a MiniLM forward pass this fence wait
 * roughly 80–100 times per text – that is the dominant cost in
 * BERT-style encoder pipelines (≈ 44 ms/text in the baseline benchmark).
 * <p>
 * When a batch is active, the {@link D3D12Bindings#executeOrDefer}
 * helper instead:
 * <ul>
 *   <li>closes + submits the command list as usual,</li>
 *   <li>{@code AddRef}s both the command list and its allocator so they
 *       outlive the kernel's local {@code try/finally},</li>
 *   <li>enqueues them on this batch, and</li>
 *   <li>returns immediately without ever creating a fence.</li>
 * </ul>
 * On {@link #close()} a single fence is created, signalled and waited
 * for – draining every pending submission in one shot – then all
 * retained command lists and allocators are released.
 * <p>
 * <b>Correctness:</b> D3D12 guarantees in-order execution of command
 * lists submitted to the same queue. To ensure UAV writes from one
 * submission are visible to the next, {@code executeOrDefer} additionally
 * records a global {@code UAV_BARRIER(ALL)} as the last command of every
 * batched list. The non-batched path (a single CL followed by a fence
 * wait) does not need that barrier because the fence wait itself drains
 * the GPU.
 * <p>
 * Usage:
 * <pre>{@code
 *   try (DirectMlGpuBatch batch = DirectMlGpuBatch.begin(wb)) {
 *       // every kernel.dispatch(...) inside this block becomes
 *       // "fire-and-forget"; one fence wait happens in close().
 *       block.dispatch(xIn, w, mask, xOut);
 *   }
 * }</pre>
 * Batches are <b>re-entrant</b>: a nested {@link #begin(WindowsBindings)}
 * call on the same thread returns a no-op handle that shares state with
 * the outer batch; only the outermost {@link #close()} drains the fence
 * and releases the retained command lists.
 */
public class DirectMlGpuBatch implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlGpuBatch.class);
    private static final ThreadLocal<DirectMlGpuBatch> CURRENT = new ThreadLocal<>();

    /**
     * Cumulative count of GPU fence drains performed across all threads
     * since process start. Bumped once per {@link #close()} (regardless
     * of how many command lists were retained). Outside a batch the
     * fence drains happen inside {@link D3D12Bindings#executeAndWait}
     * and bump {@link #standaloneFenceWaits} instead.
     */
    private static final java.util.concurrent.atomic.AtomicLong batchFenceWaits =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong standaloneFenceWaits =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong batchedSubmissions =
            new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong coalescedLayerSubmissions =
            new java.util.concurrent.atomic.AtomicLong();

    private final WindowsBindings wb;
    private final List<MemorySegment> retainedCls = new ArrayList<>();
    private final List<MemorySegment> retainedAllocs = new ArrayList<>();
    private final boolean nested;
    private boolean closed = false;

    private DirectMlGpuBatch(WindowsBindings wb, boolean nested) {
        this.wb = wb;
        this.nested = nested;
    }

    /**
     * Open a submission batch on the current thread. Nested calls are
     * allowed and act as no-ops – they return the outer batch unchanged
     * and the matching {@link #close()} will not drain. Only the
     * outermost {@code begin}/{@code close} pair performs the actual
     * fence wait. Pair with try-with-resources.
     */
    public static DirectMlGpuBatch begin(WindowsBindings wb) {
        if (wb == null) throw new IllegalArgumentException("wb");
        DirectMlGpuBatch existing = CURRENT.get();
        if (existing != null) {
            // Re-entrant: return a no-op wrapper that shares state but
            // does not drain on close.
            return new DirectMlGpuBatch.NestedHandle(existing);
        }
        DirectMlGpuBatch b = new DirectMlGpuBatch(wb, false);
        CURRENT.set(b);
        return b;
    }

    /**
     * Returns the active batch on this thread, or {@code null} when none is open.
     */
    public static DirectMlGpuBatch current() {
        return CURRENT.get();
    }

    /**
     * Convenience: {@code current() != null}.
     */
    public static boolean isActive() {
        return CURRENT.get() != null;
    }

    /**
     * AddRef + enqueue a freshly submitted command list and its allocator.
     * Called from {@link D3D12Bindings#executeOrDefer}; not part of the
     * public kernel API.
     */
    public void retain(MemorySegment cmdList, MemorySegment cmdAllocator) {
        if (closed) throw new IllegalStateException("DirectMlGpuBatch already closed");
        DxgiBindings.addRef(cmdList);
        DxgiBindings.addRef(cmdAllocator);
        retainedCls.add(cmdList);
        retainedAllocs.add(cmdAllocator);
        batchedSubmissions.incrementAndGet();
    }

    /**
     * Submissions currently retained inside this batch.
     */
    public int submissions() {
        return retainedCls.size();
    }

    /**
     * Drain every retained submission with a single fence wait, then
     * release the held command lists and allocators. The thread-local
     * is cleared even when the wait throws.
     */
    @Override
    public void close() throws DirectMlRuntimeException {
        if (closed) return;
        closed = true;
        if (nested) {
            // Nested handle – outer batch is still active, do nothing.
            return;
        }
        CURRENT.remove();

        if (retainedCls.isEmpty()) {
            return;
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment fence = D3D12Bindings.createFence(wb.getD3d12Device(), 0, arena);
            try {
                D3D12Bindings.queueSignal(wb.getCommandQueue(), fence, 1);
                long timeoutMs = 10_000;
                long deadline = System.currentTimeMillis() + timeoutMs;
                while (D3D12Bindings.fenceGetCompletedValue(fence) < 1) {
                    if (System.currentTimeMillis() > deadline) {
                        throw new DirectMlRuntimeException(
                                "DirectMlGpuBatch fence timeout after " + timeoutMs
                                        + " ms (" + retainedCls.size() + " submissions in flight)");
                    }
                    Thread.onSpinWait();
                }
                batchFenceWaits.incrementAndGet();
                // GEMMA-WARP-13b-3b: one fence wait drains every deferred submission since the last drain.
                com.aresstack.windirectml.windows.WarpSubmissionStats.recordFenceWait();
            } finally {
                DxgiBindings.release(fence);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException("DirectMlGpuBatch.close failed", e);
        } finally {
            // Release once for every AddRef we did in retain(...). The
            // kernels themselves already released their original ref in
            // their own try/finally, so this brings the refcount to 0.
            for (MemorySegment cl : retainedCls) {
                try {
                    DxgiBindings.release(cl);
                } catch (Exception ignored) {
                }
            }
            for (MemorySegment a : retainedAllocs) {
                try {
                    DxgiBindings.release(a);
                } catch (Exception ignored) {
                }
            }
            retainedCls.clear();
            retainedAllocs.clear();
        }
    }

    // ── Metrics ─────────────────────────────────────────────────────────

    /**
     * Number of fence drains performed via {@link #close()}.
     */
    public static long batchFenceWaits() {
        return batchFenceWaits.get();
    }

    /**
     * Number of fence drains performed by {@link D3D12Bindings#executeAndWait}
     * outside of any batch.
     */
    public static long standaloneFenceWaits() {
        return standaloneFenceWaits.get();
    }

    /**
     * Cumulative submissions that were coalesced under some batch.
     */
    public static long batchedSubmissions() {
        return batchedSubmissions.get();
    }

    /**
     * Cumulative number of per-encoder-layer coalesced submissions.
     * Bumped once per
     * {@code DirectMlBertEncoderLayerBlock.dispatch} – exactly once per
     * encoder layer regardless of how many sub-ops the layer contains.
     * Compare against {@link #batchedSubmissions()} (which counts every
     * raw ExecuteCommandLists call) to verify per-layer coalescing.
     */
    public static long coalescedLayerSubmissions() {
        return coalescedLayerSubmissions.get();
    }

    /**
     * Increment {@link #coalescedLayerSubmissions}. Called from the
     * encoder block at the end of its single per-layer ExecuteCommandLists.
     */
    public static void recordCoalescedLayerSubmission() {
        coalescedLayerSubmissions.incrementAndGet();
    }

    /**
     * Increment {@link #standaloneFenceWaits}. Called by D3D12Bindings.
     */
    public static void recordStandaloneFenceWait() {
        standaloneFenceWaits.incrementAndGet();
    }

    /**
     * Inner no-op handle returned for nested {@link #begin(WindowsBindings)}
     * calls. Forwards {@link #retain} to the outer batch and does nothing
     * on {@link #close}.
     */
    private static final class NestedHandle extends DirectMlGpuBatch {
        private final DirectMlGpuBatch outer;

        NestedHandle(DirectMlGpuBatch outer) {
            super(outer.wb, true);
            this.outer = outer;
        }

        @Override
        public void retain(MemorySegment cmdList, MemorySegment cmdAllocator) {
            outer.retain(cmdList, cmdAllocator);
        }

        @Override
        public int submissions() {
            return outer.submissions();
        }

        @Override
        public void close() { /* no-op: outer owns the drain */ }
    }
}

