package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WarpSubmissionStats;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-13b-2: the GPU-resident kernel seam produces the same numbers as the old float[] API and
 * does fewer CPU readbacks (intermediates stay on the GPU). Synthetic + device-gated; runs in the
 * default suite.
 */
@EnabledOnOs(OS.WINDOWS)
class Gemma3WarpResidentPipelineTest {

    private static final float ABS_TOL = 1e-4f;
    private static final float REL_TOL = 1e-4f;

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
    void rmsNormThenGeGluResidentMatchesFloatPathWithFewerReadbacks() throws Exception {
        int inter = 16;
        int dim = 2 * inter;
        Random rng = new Random(5);
        float[] x = rand(rng, dim);       // gate|up vector to normalise then GeGLU
        float[] w = rand(rng, dim);
        float eps = 1e-6f;

        try (Gemma3WarpRmsNormKernel rms = new Gemma3WarpRmsNormKernel(wb);
             Gemma3WarpGeGluKernel geglu = new Gemma3WarpGeGluKernel(wb)) {

            // float[] path (old, one readback per kernel)
            WarpSubmissionStats.reset();
            WarpSubmissionStats.Snapshot f0 = WarpSubmissionStats.snapshot();
            float[] normedF = rms.normalize(x, w, eps);
            float[] outF = geglu.apply(normedF, inter);
            long floatReadbacks = WarpSubmissionStats.snapshot().minus(f0).readbacks();

            // resident path (intermediates stay on the GPU, single final readback)
            WarpExecutionContext ctx = new WarpExecutionContext(wb);
            WarpSubmissionStats.Snapshot r0 = WarpSubmissionStats.snapshot();
            float[] outR;
            try (WarpGpuBuffer xb = ctx.upload(x);
                 WarpGpuBuffer wbuf = ctx.upload(w);
                 WarpGpuBuffer normed = rms.normalize(ctx, xb, wbuf, eps);
                 WarpGpuBuffer out = geglu.apply(ctx, normed, inter)) {
                outR = out.readback();
            }
            long residentReadbacks = WarpSubmissionStats.snapshot().minus(r0).readbacks();

            assertClose(outF, outR);
            System.out.println("[13b-2] rms->geglu floatReadbacks=" + floatReadbacks
                    + " residentReadbacks=" + residentReadbacks);
            assertTrue(residentReadbacks < floatReadbacks,
                    "resident pipeline should do fewer readbacks: resident=" + residentReadbacks
                            + " float=" + floatReadbacks);
            assertEquals(1, residentReadbacks, "only the final result is read back");
        }
    }

    @Test
    void attentionChainResidentMatchesFloatPathWithFewerReadbacks() throws Exception {
        int numHeads = 2;
        int numKvHeads = 1;
        int headDim = 4;
        int kvDim = numKvHeads * headDim;
        int seqLen = 3;
        int queryPos = 2;
        int firstValid = 0;
        int pos = 2;
        float theta = 10_000f;
        float scale = 0.5f;
        Random rng = new Random(7);
        float[] q = rand(rng, numHeads * headDim);
        float[] keys = rand(rng, seqLen * kvDim);
        float[] values = rand(rng, seqLen * kvDim);

        try (Gemma3WarpRoPEKernel rope = new Gemma3WarpRoPEKernel(wb);
             Gemma3WarpAttentionScoresKernel scores = new Gemma3WarpAttentionScoresKernel(wb);
             Gemma3WarpSoftmaxKernel softmax = new Gemma3WarpSoftmaxKernel(wb);
             Gemma3WarpAttentionValueKernel value = new Gemma3WarpAttentionValueKernel(wb)) {

            // float[] path: RoPE(q) -> scores -> softmax -> value (one readback each)
            WarpSubmissionStats.Snapshot f0 = WarpSubmissionStats.snapshot();
            float[] qRopeF = rope.applyToHeads(q, numHeads, headDim, pos, theta);
            float[] scoresF = scores.scores(qRopeF, keys, numHeads, numKvHeads, headDim, seqLen, queryPos, firstValid, scale);
            float[] probF = softmax.softmaxRows(scoresF, numHeads, seqLen);
            float[] ctxF = value.aggregate(probF, values, numHeads, numKvHeads, headDim, seqLen);
            long floatReadbacks = WarpSubmissionStats.snapshot().minus(f0).readbacks();

            // resident path: same chain, intermediates GPU-resident, single final readback
            WarpExecutionContext ctx = new WarpExecutionContext(wb);
            WarpSubmissionStats.Snapshot r0 = WarpSubmissionStats.snapshot();
            float[] ctxR;
            try (WarpGpuBuffer qb = ctx.upload(q);
                 WarpGpuBuffer keysB = ctx.upload(keys);
                 WarpGpuBuffer valuesB = ctx.upload(values);
                 WarpGpuBuffer qRope = rope.applyToHeads(ctx, qb, numHeads, headDim, pos, theta);
                 WarpGpuBuffer sc = scores.scores(ctx, qRope, keysB, numHeads, numKvHeads, headDim, seqLen, queryPos, firstValid, scale);
                 WarpGpuBuffer prob = softmax.softmaxRows(ctx, sc, numHeads, seqLen);
                 WarpGpuBuffer out = value.aggregate(ctx, prob, valuesB, numHeads, numKvHeads, headDim, seqLen)) {
                ctxR = out.readback();
            }
            long residentReadbacks = WarpSubmissionStats.snapshot().minus(r0).readbacks();

            assertClose(ctxF, ctxR);
            System.out.println("[13b-2] attn-chain floatReadbacks=" + floatReadbacks
                    + " residentReadbacks=" + residentReadbacks);
            assertTrue(residentReadbacks < floatReadbacks,
                    "resident chain should do fewer readbacks: resident=" + residentReadbacks
                            + " float=" + floatReadbacks);
            assertEquals(1, residentReadbacks, "only the final context is read back");
        }
    }

    private static float[] rand(Random rng, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rng.nextFloat() * 2 - 1;
        }
        return v;
    }

    private static void assertClose(float[] want, float[] got) {
        assertEquals(want.length, got.length, "length");
        for (int i = 0; i < want.length; i++) {
            float tol = ABS_TOL + REL_TOL * Math.abs(want[i]);
            assertEquals(want[i], got[i], tol, "element[" + i + "]");
        }
    }
}
