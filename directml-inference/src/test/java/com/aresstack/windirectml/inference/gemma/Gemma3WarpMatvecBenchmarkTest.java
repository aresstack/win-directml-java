package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.MatMulNBitsKernel;
import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.Locale;
import java.util.Random;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GEMMA-WARP-14a: compute benchmark + decision for the Gemma matvec/projection on this host — DML
 * MatMulNBits (FP32 GEMM) vs the custom WARP FP32 matvec shader vs the custom INT4 matvec shader — for the
 * real Gemma 3 270M projection shapes (decode, M=1). 13g showed decode is compute-bound (not submit-bound),
 * so this measures which matvec implementation is fastest and projects a decode-token estimate to pick the
 * direction. Measurement only: no production change, no flaky assertions; gated opt-in via
 * {@code -Dgemma.warp.bench=true} (heavy — builds many kernels + uploads the LM-head weight).
 */
@EnabledOnOs(OS.WINDOWS)
@EnabledIfSystemProperty(named = "gemma.warp.bench", matches = "true")
class Gemma3WarpMatvecBenchmarkTest {

    private static final int BLOCK = 128;
    private static final int WARMUP = 5;
    private static final int ITERS = 25;
    private static final int LMHEAD_ITERS = 5;

    /** Gemma 3 270M projection shapes [N=out, K=in] and how many run per decode token (18 layers). */
    private record Shape(String name, int n, int k, int perToken) {
    }

    private static final Shape[] SHAPES = {
            new Shape("q_proj   ", 1024, 640, 18),
            new Shape("k_proj   ", 256, 640, 18),
            new Shape("v_proj   ", 256, 640, 18),
            new Shape("o_proj   ", 640, 1024, 18),
            new Shape("gate_proj", 2048, 640, 18),
            new Shape("up_proj  ", 2048, 640, 18),
            new Shape("down_proj", 640, 2048, 18),
            new Shape("lm_head  ", 262144, 640, 1),
    };

    @Test
    void benchmarkMatvecImplementations() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML device");
        WindowsBindings warp = new WindowsBindings();
        warp.init("warp");
        WindowsBindings dml = new WindowsBindings();
        dml.init("directml");
        Random rng = new Random(140);

        System.out.println("[14a] Gemma matvec benchmark (M=1 decode, ms/call; lower is better)");
        System.out.printf(Locale.ROOT, "[14a] %-9s %6s %6s   %10s %10s %10s%n",
                "shape", "N", "K", "DML-GEMM", "WARP-FP32", "INT4-WARP");

        double[] perTokenMs = new double[3]; // DML, WARP-FP32, INT4
        for (Shape s : SHAPES) {
            int iters = s.n >= 100_000 ? LMHEAD_ITERS : ITERS;
            double dmlMs = time(() -> buildFp32(dml, s, rng), iters);
            double warpFp32Ms = time(() -> buildFp32(warp, s, rng), iters);
            double int4Ms = time(() -> buildInt4(warp, s, rng), iters);
            System.out.printf(Locale.ROOT, "[14a] %-9s %6d %6d   %10.3f %10.3f %10.3f%n",
                    s.name.trim(), s.n, s.k, dmlMs, warpFp32Ms, int4Ms);
            perTokenMs[0] += dmlMs * s.perToken;
            perTokenMs[1] += warpFp32Ms * s.perToken;
            perTokenMs[2] += int4Ms * s.perToken;
        }

        System.out.printf(Locale.ROOT,
                "[14a] estimated matvec ms/decode-token (18 layers ×7 proj + lm_head): "
                        + "DML-GEMM=%.1f  WARP-FP32=%.1f  INT4-WARP=%.1f%n",
                perTokenMs[0], perTokenMs[1], perTokenMs[2]);
        String[] names = {"DML-GEMM", "WARP-FP32", "INT4-WARP"};
        int best = 0;
        for (int i = 1; i < 3; i++) {
            if (perTokenMs[i] > 0 && (perTokenMs[best] <= 0 || perTokenMs[i] < perTokenMs[best])) {
                best = i;
            }
        }
        System.out.println("[14a] DECISION: fastest matvec for Gemma decode on this host = " + names[best]
                + " (current product path = DML-GEMM via -Dgemma backend=directml)");

        warp.close();
        dml.close();
    }

    /** A kernel + its resident in/out buffers, timed together and closed together. */
    private interface KernelSetup {
        MatMulNBitsKernel kernel();

        WarpExecutionContext ctx();

        WarpGpuBuffer in();

        WarpGpuBuffer out();

        void close();
    }

    private double time(java.util.function.Supplier<KernelSetup> factory, int iters) {
        KernelSetup s = null;
        try {
            s = factory.get();
            for (int i = 0; i < WARMUP; i++) {
                s.kernel().matvecResident(s.in(), s.out());
            }
            long t = System.nanoTime();
            for (int i = 0; i < iters; i++) {
                s.kernel().matvecResident(s.in(), s.out());
            }
            return (System.nanoTime() - t) / 1e6 / iters;
        } catch (Throwable t) {
            System.out.println("[14a]   (impl unavailable for this shape: "
                    + (t.getMessage() != null ? t.getMessage() : t.getClass().getSimpleName()) + ")");
            return -1.0;
        } finally {
            if (s != null) {
                s.close();
            }
        }
    }

    private KernelSetup buildFp32(WindowsBindings wb, Shape s, Random rng) {
        float[] w = new float[s.n * s.k];
        for (int i = 0; i < w.length; i++) {
            w[i] = (rng.nextFloat() * 2 - 1) * 0.05f;
        }
        MatMulNBitsKernel kernel = MatMulNBitsKernel.fromDequantizedWeights(wb, s.n, s.k, w);
        return setup(wb, kernel, s, rng);
    }

    private KernelSetup buildInt4(WindowsBindings wb, Shape s, Random rng) {
        int blocksPerRow = s.k / BLOCK;
        byte[] qWeight = new byte[s.n * (s.k / 2)];
        rng.nextBytes(qWeight);
        byte[] zp = new byte[(s.n * blocksPerRow + 1) / 2];
        rng.nextBytes(zp);
        float[] scales = new float[s.n * blocksPerRow];
        for (int i = 0; i < scales.length; i++) {
            scales[i] = 0.01f + rng.nextFloat() * 0.02f;
        }
        MatMulNBitsKernel kernel = new MatMulNBitsKernel(wb, s.n, s.k, qWeight, scales, zp, BLOCK);
        return setup(wb, kernel, s, rng);
    }

    private KernelSetup setup(WindowsBindings wb, MatMulNBitsKernel kernel, Shape s, Random rng) {
        try {
            WarpExecutionContext ctx = new WarpExecutionContext(wb);
            float[] x = new float[s.k];
            for (int i = 0; i < x.length; i++) {
                x[i] = (rng.nextFloat() * 2 - 1) * 0.1f;
            }
            WarpGpuBuffer in = ctx.upload(x);
            WarpGpuBuffer out = WarpGpuBuffer.allocate(wb, s.n);
            return new KernelSetup() {
                @Override public MatMulNBitsKernel kernel() { return kernel; }
                @Override public WarpExecutionContext ctx() { return ctx; }
                @Override public WarpGpuBuffer in() { return in; }
                @Override public WarpGpuBuffer out() { return out; }
                @Override public void close() {
                    in.close();
                    out.close();
                    kernel.close();
                }
            };
        } catch (Exception e) {
            kernel.close();
            throw new RuntimeException(e);
        }
    }
}
