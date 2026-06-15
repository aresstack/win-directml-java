package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.warp.WarpDenseProjection;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.util.Objects;

/**
 * One full Gemma 3 transformer layer on WARP/DirectML (GEMMA-WARP-8), composed from the validated
 * building blocks and checked against {@code Gemma3ReferenceForwardPass.runLayer}.
 *
 * <p>Gemma layer structure (kept distinct from Qwen/SmolLM2): for each position
 * {@code input → zero-centered RMSNorm → q/k/v proj → QK-norm → dual-theta RoPE → GQA attention with
 * the local/global sliding-window + causal mask → o_proj → post-attention RMSNorm → +residual →
 * pre-feedforward RMSNorm → GeGLU MLP → post-feedforward RMSNorm → +residual}. The visible key range
 * and per-layer theta/scale come from {@link Gemma3AttentionLayout}.</p>
 *
 * <p>The stateless compute kernels are supplied as a shared {@link Gemma3WarpKernels} bundle (built once
 * across all layers); only the per-layer projections are owned here, so the resident kernel count stays
 * bounded at full-model scale. A building-block composition (CPU readback between kernels), the parity
 * step before the fused single-submission product pipeline. Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpLayer implements AutoCloseable {

    private final int layer;
    private final int hidden;
    private final int numHeads;
    private final int numKvHeads;
    private final int headDim;
    private final int kvDim;
    private final float eps;
    private final float theta;
    private final float scale;
    private final Gemma3AttentionLayout layout;

    private final float[] inputLayerNorm;
    private final float[] qNorm;
    private final float[] kNorm;
    private final float[] postAttentionLayerNorm;
    private final float[] preFeedforwardLayerNorm;
    private final float[] postFeedforwardLayerNorm;

    private final Gemma3WarpKernels k;
    private final boolean ownsKernels;
    private final WarpDenseProjection qProj;
    private final WarpDenseProjection kProj;
    private final WarpDenseProjection vProj;
    private final WarpDenseProjection oProj;
    private final Gemma3WarpMlp mlp;
    private boolean closed;

    /** Self-contained layer that builds and owns its compute kernels (standalone use / tests). */
    public Gemma3WarpLayer(WindowsBindings wb, Gemma3Config config, int layerIndex,
                           Gemma3WarpLayerWeights w) throws WindowsNativeException {
        this(wb, config, layerIndex, w, new Gemma3WarpKernels(wb), true);
    }

    /** Layer using a <b>shared</b> {@link Gemma3WarpKernels} bundle (not owned/closed here). */
    public Gemma3WarpLayer(WindowsBindings wb, Gemma3Config config, int layerIndex,
                           Gemma3WarpLayerWeights w, Gemma3WarpKernels shared) throws WindowsNativeException {
        this(wb, config, layerIndex, w, Objects.requireNonNull(shared, "shared"), false);
    }

    private Gemma3WarpLayer(WindowsBindings wb, Gemma3Config config, int layerIndex,
                            Gemma3WarpLayerWeights w, Gemma3WarpKernels kernels, boolean ownsKernels)
            throws WindowsNativeException {
        Objects.requireNonNull(wb, "wb");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(w, "w");
        this.layer = layerIndex;
        this.hidden = config.hiddenSize();
        this.numHeads = config.numAttentionHeads();
        this.numKvHeads = config.numKeyValueHeads();
        this.headDim = config.headDim();
        this.kvDim = config.keyValueDim();
        this.eps = (float) config.rmsNormEps();
        this.layout = new Gemma3AttentionLayout(config);
        this.theta = (float) layout.ropeTheta(layerIndex);
        this.scale = layout.attentionScale();

        this.inputLayerNorm = w.inputLayerNorm();
        this.qNorm = w.qNorm();
        this.kNorm = w.kNorm();
        this.postAttentionLayerNorm = w.postAttentionLayerNorm();
        this.preFeedforwardLayerNorm = w.preFeedforwardLayerNorm();
        this.postFeedforwardLayerNorm = w.postFeedforwardLayerNorm();

        this.k = kernels;
        this.ownsKernels = ownsKernels;
        int attnDim = config.attentionDim();
        this.qProj = WarpDenseProjection.fromDequantizedWeights(wb, "gemma3.q_proj", attnDim, hidden, w.qProj());
        this.kProj = WarpDenseProjection.fromDequantizedWeights(wb, "gemma3.k_proj", kvDim, hidden, w.kProj());
        this.vProj = WarpDenseProjection.fromDequantizedWeights(wb, "gemma3.v_proj", kvDim, hidden, w.vProj());
        this.oProj = WarpDenseProjection.fromDequantizedWeights(wb, "gemma3.o_proj", hidden, attnDim, w.oProj());
        this.mlp = new Gemma3WarpMlp(wb, hidden, config.intermediateSize(),
                w.gateProj(), w.upProj(), w.downProj(), kernels.geGlu());
    }

    /**
     * Run the layer over a {@code [seqLen][hidden]} state, returning the same array mutated in place
     * (matching the reference's residual semantics).
     */
    public float[][] forward(float[][] state) throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(state, "state");
        int s = state.length;

        // 1. q/k/v projections + QK-norm + RoPE for every position.
        float[][] q = new float[s][];
        float[] kFlat = new float[s * kvDim];
        float[] vFlat = new float[s * kvDim];
        for (int t = 0; t < s; t++) {
            float[] normed = k.rmsNorm().normalize(state[t], inputLayerNorm, eps);
            float[] qt = qProj.project(normed);
            float[] kt = kProj.project(normed);
            float[] vt = vProj.project(normed);
            qt = k.qkNorm().normalizeHeads(qt, numHeads, headDim, qNorm, eps);
            kt = k.qkNorm().normalizeHeads(kt, numKvHeads, headDim, kNorm, eps);
            qt = k.rope().applyToHeads(qt, numHeads, headDim, t, theta);
            kt = k.rope().applyToHeads(kt, numKvHeads, headDim, t, theta);
            q[t] = qt;
            System.arraycopy(kt, 0, kFlat, t * kvDim, kvDim);
            System.arraycopy(vt, 0, vFlat, t * kvDim, kvDim);
        }

        // 2. Attention -> o_proj -> post-attention norm -> residual (attention reads q/k/v, not state).
        for (int t = 0; t < s; t++) {
            int firstValid = layout.firstValidKey(layer, t);
            float[] scores = k.scores().scores(q[t], kFlat, numHeads, numKvHeads, headDim, s, t, firstValid, scale);
            float[] prob = k.softmax().softmaxRows(scores, numHeads, s);
            float[] context = k.value().aggregate(prob, vFlat, numHeads, numKvHeads, headDim, s);
            float[] attnProj = oProj.project(context);
            attnProj = k.rmsNorm().normalize(attnProj, postAttentionLayerNorm, eps);
            for (int i = 0; i < hidden; i++) {
                state[t][i] += attnProj[i];
            }
        }

        // 3. GeGLU MLP with pre/post feedforward sandwich norms + residual.
        for (int t = 0; t < s; t++) {
            float[] ff = k.rmsNorm().normalize(state[t], preFeedforwardLayerNorm, eps);
            float[] down = mlp.mlp(ff);
            down = k.rmsNorm().normalize(down, postFeedforwardLayerNorm, eps);
            for (int i = 0; i < hidden; i++) {
                state[t][i] += down[i];
            }
        }
        return state;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpLayer is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            qProj.close();
            kProj.close();
            vProj.close();
            oProj.close();
            mlp.close();
            if (ownsKernels) {
                k.close();
            }
        }
    }
}
