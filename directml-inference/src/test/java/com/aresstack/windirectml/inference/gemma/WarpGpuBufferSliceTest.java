package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-17 hardening for {@link WarpGpuBuffer#slice} (the zero-copy view used to split the fused
 * QKV/GateUp matmul output): bounds/offset checks, the 4-byte-aligned VA offset, non-owning close()
 * semantics, and a functional check that a slice binds correctly as a kernel UAV input.
 *
 * <p>Skipped (assumption-aborted) when no DirectML/D3D12 device is present.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class WarpGpuBufferSliceTest {

    private static WindowsBindings wb;

    @BeforeAll
    static void initGpu() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        wb = new WindowsBindings();
        wb.init("directml");
    }

    @AfterAll
    static void closeGpu() {
        if (wb != null) {
            wb.close();
        }
    }

    @Test
    void sliceBindsAtTheOffsetAndComputesCorrectly() throws Exception {
        int m = 320;
        float[] data = new float[2 * m];
        for (int i = 0; i < data.length; i++) {
            data[i] = i * 0.5f - 7.0f;
        }
        WarpExecutionContext ctx = new WarpExecutionContext(wb);
        try (WarpGpuBuffer parent = WarpGpuBuffer.upload(wb, data);
             Gemma3WarpElementAddKernel add = new Gemma3WarpElementAddKernel(wb)) {
            WarpGpuBuffer a = parent.slice(0, m);
            WarpGpuBuffer b = parent.slice(m, m);
            // VA offset is exactly elementOffset*4 bytes from the base.
            assertEquals(parent.gpuAddress(), a.gpuAddress(), "slice(0,..) starts at the base address");
            assertEquals(parent.gpuAddress() + (long) m * Float.BYTES, b.gpuAddress(),
                    "slice(m,..) is offset by m*4 bytes");
            assertEquals(m, a.elementCount());

            try (WarpGpuBuffer out = add.add(ctx, a, b)) {
                float[] result = out.readback();
                for (int i = 0; i < m; i++) {
                    assertEquals(data[i] + data[m + i], result[i], 1e-4f, "sum[" + i + "]");
                }
            }
        }
    }

    @Test
    void boundsAndOffsetChecks() throws Exception {
        try (WarpGpuBuffer parent = WarpGpuBuffer.allocate(wb, 16)) {
            assertThrows(IllegalArgumentException.class, () -> parent.slice(-1, 4), "negative offset");
            assertThrows(IllegalArgumentException.class, () -> parent.slice(0, 0), "zero count");
            assertThrows(IllegalArgumentException.class, () -> parent.slice(13, 4), "offset+count > size");
            assertThrows(IllegalArgumentException.class, () -> parent.slice(0, 17), "count > size");
        }
    }

    @Test
    void aViewCannotBeReslicedReadBackAndCloseIsNonOwning() throws Exception {
        try (WarpGpuBuffer parent = WarpGpuBuffer.allocate(wb, 16)) {
            WarpGpuBuffer view = parent.slice(4, 8);
            assertThrows(IllegalStateException.class, () -> view.slice(0, 2), "cannot slice a view");
            assertThrows(IllegalStateException.class, view::readback, "cannot read back a view");
            // Closing the view must NOT release the parent's buffer.
            view.close();
            float[] parentData = parent.readback(); // parent still valid after the view was closed
            assertEquals(16, parentData.length);
        }
    }
}
