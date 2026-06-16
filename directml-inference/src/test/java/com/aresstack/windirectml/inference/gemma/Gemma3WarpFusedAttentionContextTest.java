package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-14b: the fused attention-context kernel equals the staged scores→softmax→value path within
 * fp32 tolerance, for the Gemma layouts — GQA 4/1, head_dim=256, full and local (sliding-window) masks.
 * Tolerance abs 1e-3 + rel 1e-3 (the fused online softmax accumulates in a different order).
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpFusedAttentionContextTest {

    private static final float ABS_TOL = 1e-3f;
    private static final float REL_TOL = 1e-3f;

    private static WindowsBindings wb;
    private static Gemma3WarpKernels kernels;

    @BeforeAll
    static void init() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        wb = new WindowsBindings();
        wb.init("directml");
        kernels = new Gemma3WarpKernels(wb);
    }

    @AfterAll
    static void close() {
        if (kernels != null) {
            kernels.close();
        }
        if (wb != null) {
            wb.close();
        }
    }

    @Test
    void gqa4x1HeadDim256FullMask() throws Exception {
        check(4, 1, 256, 6, 0, new Random(1));     // Gemma 270M shape, full attention
    }

    @Test
    void gqa4x1HeadDim256LocalWindow() throws Exception {
        check(4, 1, 256, 12, 7, new Random(2));    // local layer: firstValid > 0 (sliding window)
    }

    @Test
    void noGqaSmallHeadDim() throws Exception {
        check(2, 2, 8, 5, 0, new Random(3));       // numKvHeads == numHeads
    }

    @Test
    void gqa4x2LocalMidHeadDim() throws Exception {
        check(4, 2, 16, 8, 3, new Random(4));
    }

    @Test
    void singlePosition() throws Exception {
        check(4, 1, 256, 1, 0, new Random(5));     // queryPos=0, one visible key
    }

    private void check(int numHeads, int numKvHeads, int headDim, int seqLen, int firstValid, Random rng)
            throws Exception {
        int kvDim = numKvHeads * headDim;
        int queryPos = seqLen - 1;
        float scale = (float) (1.0 / Math.sqrt(headDim));

        WarpExecutionContext ctx = new WarpExecutionContext(wb);
        WarpGpuBuffer q = ctx.upload(rand(rng, numHeads * headDim, 0.1f));
        WarpGpuBuffer keys = ctx.upload(rand(rng, seqLen * kvDim, 0.1f));
        WarpGpuBuffer values = ctx.upload(rand(rng, seqLen * kvDim, 0.1f));

        // staged
        WarpGpuBuffer scores = kernels.scores().scores(ctx, q, keys, numHeads, numKvHeads, headDim,
                seqLen, queryPos, firstValid, scale);
        WarpGpuBuffer prob = kernels.softmax().softmaxRows(ctx, scores, numHeads, seqLen);
        WarpGpuBuffer stagedCtx = kernels.value().aggregate(ctx, prob, values, numHeads, numKvHeads, headDim, seqLen);
        float[] staged = stagedCtx.readback();

        // fused
        WarpGpuBuffer fusedCtx = kernels.fusedAttention().context(ctx, q, keys, values,
                numHeads, numKvHeads, headDim, kvDim, queryPos, firstValid, scale);
        float[] fused = fusedCtx.readback();

        assertEquals(staged.length, fused.length, "context length");
        for (int i = 0; i < staged.length; i++) {
            float tol = ABS_TOL + REL_TOL * Math.abs(staged[i]);
            assertEquals(staged[i], fused[i], tol,
                    "fused != staged at [" + i + "] (heads=" + numHeads + "/" + numKvHeads + " hd=" + headDim
                            + " seq=" + seqLen + " firstValid=" + firstValid + ")");
        }

        q.close();
        keys.close();
        values.close();
        scores.close();
        prob.close();
        stagedCtx.close();
        fusedCtx.close();
    }

    private static float[] rand(Random rng, int n, float range) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rng.nextFloat() * 2 - 1) * range;
        }
        return v;
    }
}
