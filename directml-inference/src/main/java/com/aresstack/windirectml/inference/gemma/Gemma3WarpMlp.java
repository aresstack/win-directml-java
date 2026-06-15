package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.warp.WarpDenseProjection;
import com.aresstack.windirectml.inference.warp.WarpWeightSource;
import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.util.Objects;

/**
 * WARP/DirectML Gemma 3 MLP block (GEMMA-WARP-6): {@code down_proj( gelu_tanh(gate_proj·x) * up_proj·x )}.
 *
 * <p>The GPU mirror of the GeGLU MLP body in {@code Gemma3ReferenceForwardPass} (the pre/post
 * feedforward sandwich norms are <b>not</b> part of this block — those are the zero-centered RMSNorm
 * kernels from GEMMA-WARP-5a). The three matmuls reuse the shared {@link WarpDenseProjection}; the
 * GeGLU activation is {@link Gemma3WarpGeGluKernel}. Weights are row-major {@code [output, input]}
 * exactly as the reference {@code matvec} expects:</p>
 * <ul>
 *   <li>{@code gateProj}, {@code upProj}: {@code [intermediate, hidden]}</li>
 *   <li>{@code downProj}: {@code [hidden, intermediate]}</li>
 * </ul>
 *
 * <p>This is a validation building block (a CPU readback between projections and the activation), not
 * yet the fused single-submission product path. Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpMlp implements AutoCloseable {

    private final int hidden;
    private final int intermediate;
    private final WarpDenseProjection gateProj;
    private final WarpDenseProjection upProj;
    private final WarpDenseProjection downProj;
    private final Gemma3WarpGeGluKernel geGlu;
    private final boolean ownsGeGlu;
    private boolean closed;

    /**
     * Self-contained MLP that owns its GeGLU kernel.
     *
     * @param wb           initialised bindings (device up)
     * @param hidden       hidden width
     * @param intermediate MLP intermediate width
     * @param gateProj     row-major {@code [intermediate, hidden]}
     * @param upProj       row-major {@code [intermediate, hidden]}
     * @param downProj     row-major {@code [hidden, intermediate]}
     */
    public Gemma3WarpMlp(WindowsBindings wb, int hidden, int intermediate,
                         float[] gateProj, float[] upProj, float[] downProj) throws WindowsNativeException {
        this(wb, hidden, intermediate, gateProj, upProj, downProj, new Gemma3WarpGeGluKernel(wb), true);
    }

    /**
     * MLP using a <b>shared</b> GeGLU kernel (not owned/closed here) — used by the full model so the
     * stateless GeGLU kernel is built once across all layers rather than per layer.
     */
    public Gemma3WarpMlp(WindowsBindings wb, int hidden, int intermediate,
                         float[] gateProj, float[] upProj, float[] downProj,
                         Gemma3WarpGeGluKernel sharedGeGlu) throws WindowsNativeException {
        this(wb, hidden, intermediate,
                arraySource("gemma3.gate_proj", intermediate, hidden, gateProj),
                arraySource("gemma3.up_proj", intermediate, hidden, upProj),
                arraySource("gemma3.down_proj", hidden, intermediate, downProj),
                Objects.requireNonNull(sharedGeGlu, "sharedGeGlu"), false);
    }

    /**
     * MLP from {@link WarpWeightSource}s (heap-light when they wrap direct FP32 ByteBuffers) using a
     * <b>shared</b> GeGLU kernel.
     */
    public Gemma3WarpMlp(WindowsBindings wb, int hidden, int intermediate,
                         WarpWeightSource gateProj, WarpWeightSource upProj, WarpWeightSource downProj,
                         Gemma3WarpGeGluKernel sharedGeGlu) throws WindowsNativeException {
        this(wb, hidden, intermediate, gateProj, upProj, downProj,
                Objects.requireNonNull(sharedGeGlu, "sharedGeGlu"), false);
    }

    private Gemma3WarpMlp(WindowsBindings wb, int hidden, int intermediate,
                          float[] gateProj, float[] upProj, float[] downProj,
                          Gemma3WarpGeGluKernel geGlu, boolean ownsGeGlu) throws WindowsNativeException {
        this(wb, hidden, intermediate,
                arraySource("gemma3.gate_proj", intermediate, hidden, gateProj),
                arraySource("gemma3.up_proj", intermediate, hidden, upProj),
                arraySource("gemma3.down_proj", hidden, intermediate, downProj),
                geGlu, ownsGeGlu);
    }

    private Gemma3WarpMlp(WindowsBindings wb, int hidden, int intermediate,
                          WarpWeightSource gateProj, WarpWeightSource upProj, WarpWeightSource downProj,
                          Gemma3WarpGeGluKernel geGlu, boolean ownsGeGlu) throws WindowsNativeException {
        Objects.requireNonNull(wb, "wb");
        if (hidden < 1 || intermediate < 1) {
            throw new IllegalArgumentException("hidden and intermediate must be positive: hidden="
                    + hidden + ", intermediate=" + intermediate);
        }
        this.hidden = hidden;
        this.intermediate = intermediate;
        this.gateProj = WarpDenseProjection.fromWeightSource(wb, gateProj);
        this.upProj = WarpDenseProjection.fromWeightSource(wb, upProj);
        this.downProj = WarpDenseProjection.fromWeightSource(wb, downProj);
        this.geGlu = geGlu;
        this.ownsGeGlu = ownsGeGlu;
    }

    private static WarpWeightSource arraySource(String name, int outputRows, int inputColumns, float[] arr) {
        Objects.requireNonNull(arr, name);
        return WarpWeightSource.of(name, outputRows, inputColumns, null, () -> arr);
    }

    /**
     * Run the MLP for one (already pre-feedforward-normed) hidden vector {@code x} of length
     * {@code hidden}; returns the {@code down} output of length {@code hidden} (pre post-norm).
     */
    public float[] mlp(float[] x) throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(x, "x");
        if (x.length != hidden) {
            throw new IllegalArgumentException("x length must equal hidden: x=" + x.length + ", hidden=" + hidden);
        }
        float[] gate = gateProj.project(x);
        float[] up = upProj.project(x);
        float[] gateUp = new float[2 * intermediate];
        System.arraycopy(gate, 0, gateUp, 0, intermediate);
        System.arraycopy(up, 0, gateUp, intermediate, intermediate);
        float[] activated = geGlu.apply(gateUp, intermediate);
        return downProj.project(activated);
    }

    /**
     * GPU-resident MLP (GEMMA-WARP-13b-3a): {@code down_proj( gelu_tanh(gate_proj·x) * up_proj·x )} fully
     * on the GPU — gate/up come from resident projections and feed the two-buffer GeGLU, no host concat
     * or intermediate readback. {@code x} is the pre-feedforward-normed resident hidden; returns the
     * resident {@code down} output (pre post-norm). Same math as {@link #mlp(float[])}.
     */
    public WarpGpuBuffer mlp(WarpExecutionContext ctx, WarpGpuBuffer x) throws WindowsNativeException {
        // Standalone (no active batch): synchronous projections, so the gate/up/activated scratch can be
        // freed immediately. Inside a DirectMlGpuBatch use {@link #mlp(WarpExecutionContext, WarpGpuBuffer,
        // java.util.List)} so the deferred dispatches' inputs survive until the batch drains.
        java.util.List<WarpGpuBuffer> scratch = new java.util.ArrayList<>();
        WarpGpuBuffer down = mlp(ctx, x, scratch);
        for (WarpGpuBuffer b : scratch) {
            b.close();
        }
        return down;
    }

    /**
     * Resident MLP that registers its {@code gate}/{@code up}/{@code activated} intermediates in
     * {@code scratch} instead of closing them (GEMMA-WARP-13b-3b): under a {@code DirectMlGpuBatch} the
     * projections/GeGLU are deferred, so these buffers must outlive this call until the batch drains.
     * The caller closes {@code scratch} after the drain. Returns the resident {@code down} output.
     */
    public WarpGpuBuffer mlp(WarpExecutionContext ctx, WarpGpuBuffer x, java.util.List<WarpGpuBuffer> scratch)
            throws WindowsNativeException {
        ensureOpen();
        if (x.elementCount() != hidden) {
            throw new IllegalArgumentException("x length must equal hidden: x=" + x.elementCount() + ", hidden=" + hidden);
        }
        WarpGpuBuffer gate = gateProj.forwardResident(ctx, x);
        scratch.add(gate);
        WarpGpuBuffer up = upProj.forwardResident(ctx, x);
        scratch.add(up);
        WarpGpuBuffer activated = geGlu.apply(ctx, gate, up, intermediate);
        scratch.add(activated);
        return downProj.forwardResident(ctx, activated);
    }

    public int hidden() {
        return hidden;
    }

    public int intermediate() {
        return intermediate;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpMlp is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            gateProj.close();
            upProj.close();
            downProj.close();
            if (ownsGeGlu) {
                geGlu.close();
            }
        }
    }
}
