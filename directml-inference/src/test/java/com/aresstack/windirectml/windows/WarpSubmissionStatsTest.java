package com.aresstack.windirectml.windows;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * GEMMA-WARP-13b-1: the submit/fence/readback counters are pure instrumentation — device-free unit test
 * of the counting + snapshot/diff arithmetic. (Lives in the inference test module, which resolves
 * junit-platform-launcher offline; the production type is in directml-windows-bindings.)
 */
class WarpSubmissionStatsTest {

    @Test
    void countsAndSnapshotsDiffCorrectly() {
        WarpSubmissionStats.reset();
        WarpSubmissionStats.Snapshot zero = WarpSubmissionStats.snapshot();
        assertEquals(0, zero.submits());
        assertEquals(0, zero.fenceWaits());
        assertEquals(0, zero.readbacks());

        WarpSubmissionStats.Snapshot before = WarpSubmissionStats.snapshot();
        WarpSubmissionStats.recordSubmitAndFenceWait();
        WarpSubmissionStats.recordSubmitAndFenceWait();
        WarpSubmissionStats.recordSubmitAndFenceWait();
        WarpSubmissionStats.recordReadback();
        WarpSubmissionStats.recordReadback();
        WarpSubmissionStats.Snapshot after = WarpSubmissionStats.snapshot();

        WarpSubmissionStats.Snapshot delta = after.minus(before);
        assertEquals(3, delta.submits());
        assertEquals(3, delta.fenceWaits());
        assertEquals(2, delta.readbacks());
    }

    @Test
    void resetZeroesCounters() {
        WarpSubmissionStats.recordSubmitAndFenceWait();
        WarpSubmissionStats.recordReadback();
        WarpSubmissionStats.reset();
        WarpSubmissionStats.Snapshot s = WarpSubmissionStats.snapshot();
        assertEquals(0, s.submits());
        assertEquals(0, s.fenceWaits());
        assertEquals(0, s.readbacks());
    }
}
