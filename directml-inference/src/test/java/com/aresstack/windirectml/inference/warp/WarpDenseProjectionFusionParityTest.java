package com.aresstack.windirectml.inference.warp;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-17 hardening: a fused projection ({@link WarpDenseProjection#fromFusedWeightSources}) must be
 * numerically equivalent to running the parts as separate projections — the property the QKV (3-part) and
 * GateUp (2-part) fusions of GEMMA-WARP-16 rely on. Also checks the fail-fast guards (no silent wrong fused
 * matrix): mismatched input width and a part of the wrong size must throw, not produce garbage.
 *
 * <p>Skipped (assumption-aborted) when no DirectML/D3D12 device is present.</p>
 */
@EnabledOnOs(OS.WINDOWS)
class WarpDenseProjectionFusionParityTest {

    private static final float ABS_TOL = 1e-4f;
    private static final float REL_TOL = 1e-4f;
    private static final int HIDDEN = 640; // shared input width K

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
    void fusedQkvEqualsSeparateProjections() {
        Random rng = new Random(31);
        int attnDim = 1024; // 4 heads * 256 (Gemma 3 270M)
        int kvDim = 256;    // 1 kv head * 256
        assertFusedEqualsSeparate(rng, List.of(attnDim, kvDim, kvDim));
    }

    @Test
    void fusedGateUpEqualsSeparateProjections() {
        Random rng = new Random(37);
        int intermediate = 2048; // Gemma 3 270M
        assertFusedEqualsSeparate(rng, List.of(intermediate, intermediate));
    }

    @Test
    void mismatchedInputWidthThrows() {
        WarpWeightSource a = source("a", 8, HIDDEN, new Random(1));
        WarpWeightSource b = source("b", 8, HIDDEN + 1, new Random(2)); // different K
        assertThrows(IllegalArgumentException.class,
                () -> WarpDenseProjection.fromFusedWeightSources(wb, "bad", List.of(a, b)));
    }

    @Test
    void partOfWrongSizeThrows() {
        // Claim 8 rows but supply a float[] sized for 7 -> the fused size check must reject it.
        WarpWeightSource good = source("good", 8, HIDDEN, new Random(3));
        WarpWeightSource wrong = WarpWeightSource.of("wrong", 8, HIDDEN, null, () -> new float[7 * HIDDEN]);
        assertThrows(IllegalArgumentException.class,
                () -> WarpDenseProjection.fromFusedWeightSources(wb, "bad", List.of(good, wrong)));
    }

    private void assertFusedEqualsSeparate(Random rng, List<Integer> partRows) {
        float[] input = randomTensor(rng, HIDDEN);
        // Separate projections + a single fused projection from the same weights.
        WarpWeightSource[] sources = new WarpWeightSource[partRows.size()];
        float[][] separateOutputs = new float[partRows.size()][];
        for (int i = 0; i < partRows.size(); i++) {
            int rows = partRows.get(i);
            float[] w = randomTensor(rng, rows * HIDDEN);
            sources[i] = WarpWeightSource.of("part" + i, rows, HIDDEN, null, () -> w);
            try (WarpDenseProjection sep = WarpDenseProjection.fromDequantizedWeights(wb, "sep" + i, rows, HIDDEN, w)) {
                separateOutputs[i] = sep.project(input);
            }
        }
        try (WarpDenseProjection fused = WarpDenseProjection.fromFusedWeightSources(wb, "fused", List.of(sources))) {
            float[] fusedOut = fused.project(input);
            int offset = 0;
            for (int i = 0; i < partRows.size(); i++) {
                for (int r = 0; r < partRows.get(i); r++) {
                    assertClose("part" + i + "[" + r + "]", separateOutputs[i][r], fusedOut[offset + r]);
                }
                offset += partRows.get(i);
            }
            assertEquals(offset, fusedOut.length, "fused output length == sum of part rows");
        }
    }

    private static WarpWeightSource source(String name, int rows, int cols, Random rng) {
        float[] w = randomTensor(rng, rows * cols);
        return WarpWeightSource.of(name, rows, cols, null, () -> w);
    }

    private static float[] randomTensor(Random rng, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = rng.nextFloat() * 2 - 1;
        }
        return x;
    }

    private static void assertClose(String label, float want, float got) {
        float tol = ABS_TOL + REL_TOL * Math.abs(want);
        if (Math.abs(want - got) > tol) {
            fail(String.format("%s mismatch: want=%.6f got=%.6f (tol=%.3e)", label, want, got, tol));
        }
    }
}
