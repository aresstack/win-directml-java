package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.DxgiBindings;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-AUTO-GPU-1: DXGI adapter selection. {@code AdapterMode.WARP} binds the explicit software rasterizer;
 * {@code AdapterMode.HARDWARE} binds a non-software hardware adapter or fails with a clear message (this dev
 * host has no hardware GPU, so the HARDWARE case is expected to skip cleanly here). The adapter identity
 * (description / vendor id / device id / software flag) is always readable for logging.
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3DirectMlAdapterSelectionTest {

    @Test
    void warpModeBindsExplicitSoftwareAdapter() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("directml", WindowsBindings.AdapterMode.WARP);
            assertEquals(WindowsBindings.AdapterMode.WARP, wb.adapterMode());
            assertTrue(wb.isSoftwareAdapter(), "WARP must bind the software rasterizer");
            DxgiBindings.AdapterDesc d = wb.adapterDesc();
            assertNotNull(d, "adapter description must be readable");
            assertFalse(d.description().isBlank(), "adapter description must be non-empty: " + d);
            System.out.println("[AUTO-GPU] WARP adapter: " + d + " software=" + d.software());
        }
    }

    @Test
    void autoModeSelectsHardwareOrSkipsWithClearMessage() {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        WindowsBindings wb = new WindowsBindings();
        boolean hardwareFound;
        try {
            wb.init("directml", WindowsBindings.AdapterMode.HARDWARE);
            hardwareFound = true;
        } catch (WindowsNativeException e) {
            // No hardware GPU on this host — must be the clear, actionable message (test #8).
            assertTrue(e.getMessage() != null && e.getMessage().toLowerCase().contains("no hardware"),
                    "no-hardware path must give a clear message, got: " + e.getMessage());
            System.out.println("[AUTO-GPU] HARDWARE adapter unavailable (expected on this host): " + e.getMessage());
            wb.close();
            assumeTrue(false, "No hardware DirectML adapter on this host — AUTO/GPU comparison skipped");
            return;
        }
        try {
            assertFalse(wb.isSoftwareAdapter(), "HARDWARE must bind a non-software adapter");
            DxgiBindings.AdapterDesc d = wb.adapterDesc();
            assertNotNull(d);
            assertFalse(d.software(), "selected hardware adapter must not have the software flag: " + d);
            System.out.println("[AUTO-GPU] HARDWARE adapter: " + d);
        } finally {
            wb.close();
        }
        assertTrue(hardwareFound);
    }
}
