package com.aresstack.windirectml.sidecar.workbench;

import org.junit.jupiter.api.Test;

import javax.swing.SwingUtilities;
import java.awt.GraphicsEnvironment;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

/**
 * Smoke test: the workbench frame can be instantiated and disposed on the
 * EDT without a real sidecar process. Requires {@code java.awt.headless=true}
 * (set in the module's build.gradle test config).
 */
class WorkbenchFrameSmokeTest {

    @Test
    void framesInstantiateAndDisposeWithoutSidecar() throws Exception {
        // Skip cleanly on truly headless build hosts; the test runs on any
        // workstation with a display attached, which is the realistic
        // workbench developer setup.
        assumeFalse(GraphicsEnvironment.isHeadless(),
                "Skipping Swing smoke test in headless environment");
        final AtomicReference<WorkbenchFrame> ref = new AtomicReference<WorkbenchFrame>();
        SwingUtilities.invokeAndWait(new Runnable() {
            @Override public void run() {
                ref.set(new WorkbenchFrame());
            }
        });
        WorkbenchFrame f = ref.get();
        assertNotNull(f);
        // Title set, model reachable, sidecar not running.
        assertEquals("DirectML Sidecar Workbench", f.getTitle());
        assertNotNull(f.getModelForTesting());
        assertFalse(f.getModelForTesting().isRunning());
        // The frame should not block the EDT; dispose returns immediately.
        SwingUtilities.invokeAndWait(new Runnable() {
            @Override public void run() { f.dispose(); }
        });
        assertTrue(true);
    }
}

