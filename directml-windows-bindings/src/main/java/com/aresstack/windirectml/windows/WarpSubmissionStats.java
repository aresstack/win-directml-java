package com.aresstack.windirectml.windows;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Process-wide counters for WARP/DirectML GPU submissions, fence waits and readbacks (GEMMA-WARP-13b-1).
 *
 * <p>Pure instrumentation — counting only, no behaviour change. Every blocking command-list execution
 * ({@link D3D12Bindings#executeAndWait} and the {@code MatMulNBitsKernel} matvec single-submit path)
 * records one submit + one fence wait; every CPU readback records one readback. A caller measures a
 * region (e.g. one decode token) by taking a {@link #snapshot()} before and after and diffing — see
 * {@link Snapshot#minus}.</p>
 *
 * <p>Counters are global (all families); callers that only care about one model snapshot around that
 * model's calls. {@link #reset()} zeroes them.</p>
 */
public final class WarpSubmissionStats {

    private static final AtomicLong SUBMITS = new AtomicLong();
    private static final AtomicLong FENCE_WAITS = new AtomicLong();
    private static final AtomicLong READBACKS = new AtomicLong();
    private static final AtomicLong DISPATCHES = new AtomicLong();
    private static final AtomicLong UAV_BARRIERS = new AtomicLong();

    private WarpSubmissionStats() {
    }

    /**
     * Record one recorded compute dispatch (GEMMA-WARP-14b): bumped per {@code Dispatch} recorded into a
     * command list (element/attention kernels + the matvec dispatch), independent of how many share a submit.
     * Once dispatches share one command list, this is the meaningful per-token cost (not submits).
     */
    public static void recordDispatch() {
        DISPATCHES.incrementAndGet();
    }

    /** Record one UAV barrier recorded into a command list (GEMMA-WARP-14b). */
    public static void recordUavBarrier() {
        UAV_BARRIERS.incrementAndGet();
    }

    /** Record one command-list submission immediately followed by a blocking fence wait. */
    public static void recordSubmitAndFenceWait() {
        SUBMITS.incrementAndGet();
        FENCE_WAITS.incrementAndGet();
    }

    /**
     * Record one command-list submission with <b>no</b> fence wait (GEMMA-WARP-13b-3b): a deferred,
     * fire-and-forget submission whose fence is coalesced into a later {@code DirectMlGpuBatch} drain.
     * The matching drain records exactly one {@link #recordFenceWait()}.
     */
    public static void recordSubmit() {
        SUBMITS.incrementAndGet();
    }

    /**
     * Record one blocking fence wait with <b>no</b> submission (GEMMA-WARP-13b-3b): a
     * {@code DirectMlGpuBatch} drain that fences every deferred submission since the last drain in one
     * wait.
     */
    public static void recordFenceWait() {
        FENCE_WAITS.incrementAndGet();
    }

    /** Record one CPU readback of GPU results. */
    public static void recordReadback() {
        READBACKS.incrementAndGet();
    }

    public static void reset() {
        SUBMITS.set(0);
        FENCE_WAITS.set(0);
        READBACKS.set(0);
        DISPATCHES.set(0);
        UAV_BARRIERS.set(0);
    }

    /** Immutable snapshot of the current counters. */
    public static Snapshot snapshot() {
        return new Snapshot(SUBMITS.get(), FENCE_WAITS.get(), READBACKS.get(),
                DISPATCHES.get(), UAV_BARRIERS.get());
    }

    /** A point-in-time view of the counters; subtract two snapshots to measure a region. */
    public record Snapshot(long submits, long fenceWaits, long readbacks, long dispatches, long uavBarriers) {

        /** This snapshot minus an earlier {@code before} snapshot (the deltas over the measured region). */
        public Snapshot minus(Snapshot before) {
            return new Snapshot(submits - before.submits, fenceWaits - before.fenceWaits,
                    readbacks - before.readbacks, dispatches - before.dispatches,
                    uavBarriers - before.uavBarriers);
        }

        @Override
        public String toString() {
            return "submits=" + submits + " fenceWaits=" + fenceWaits + " readbacks=" + readbacks
                    + " dispatches=" + dispatches + " uavBarriers=" + uavBarriers;
        }
    }
}
