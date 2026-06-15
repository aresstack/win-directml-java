package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.warp.WarpDenseProjection;
import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
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
    private final int intermediate;
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
    // Lazily-uploaded resident norm weights (GEMMA-WARP-13b-3a), built on first decodeStepResident call.
    private WarpGpuBuffer inputLayerNormBuf;
    private WarpGpuBuffer qNormBuf;
    private WarpGpuBuffer kNormBuf;
    private WarpGpuBuffer postAttentionLayerNormBuf;
    private WarpGpuBuffer preFeedforwardLayerNormBuf;
    private WarpGpuBuffer postFeedforwardLayerNormBuf;
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
        int inter = config.intermediateSize();
        this.intermediate = inter;
        this.qProj = WarpDenseProjection.fromWeightSource(wb, w.qSource(attnDim, hidden));
        this.kProj = WarpDenseProjection.fromWeightSource(wb, w.kSource(kvDim, hidden));
        this.vProj = WarpDenseProjection.fromWeightSource(wb, w.vSource(kvDim, hidden));
        this.oProj = WarpDenseProjection.fromWeightSource(wb, w.oSource(hidden, attnDim));
        this.mlp = new Gemma3WarpMlp(wb, hidden, inter,
                w.gateSource(inter, hidden), w.upSource(inter, hidden), w.downSource(hidden, inter), kernels.geGlu());
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

    /**
     * Single-token decode step (GEMMA-WARP-10a): run one new token at sequence position {@code pos}
     * through this layer, appending its k/v to {@code cache} and attending over the cached history
     * (the visible range — full vs sliding-window — comes from {@link Gemma3AttentionLayout}). Returns
     * the layer output for the token (residuals applied), to be fed to the next layer.
     *
     * <p>Numerically identical to {@link #forward} for the last position of a causal sequence: the new
     * token's q attends to keys {@code [firstValid(pos), pos]}, exactly as a full-sequence pass would.</p>
     */
    public float[] decodeStep(float[] hiddenVec, int pos, Gemma3WarpKvCache cache) throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(hiddenVec, "hiddenVec");
        Objects.requireNonNull(cache, "cache");

        float[] normed = k.rmsNorm().normalize(hiddenVec, inputLayerNorm, eps);
        float[] qt = qProj.project(normed);
        float[] kt = kProj.project(normed);
        float[] vt = vProj.project(normed);
        qt = k.qkNorm().normalizeHeads(qt, numHeads, headDim, qNorm, eps);
        kt = k.qkNorm().normalizeHeads(kt, numKvHeads, headDim, kNorm, eps);
        qt = k.rope().applyToHeads(qt, numHeads, headDim, pos, theta);
        kt = k.rope().applyToHeads(kt, numKvHeads, headDim, pos, theta);

        cache.put(layer, pos, kt, vt);
        int seqLen = pos + 1;
        float[] kFlat = cache.kFlat(layer, seqLen);
        float[] vFlat = cache.vFlat(layer, seqLen);
        int firstValid = layout.firstValidKey(layer, pos);

        float[] scores = k.scores().scores(qt, kFlat, numHeads, numKvHeads, headDim, seqLen, pos, firstValid, scale);
        float[] prob = k.softmax().softmaxRows(scores, numHeads, seqLen);
        float[] context = k.value().aggregate(prob, vFlat, numHeads, numKvHeads, headDim, seqLen);
        float[] attnProj = k.rmsNorm().normalize(oProj.project(context), postAttentionLayerNorm, eps);

        float[] out = hiddenVec.clone();
        for (int i = 0; i < hidden; i++) {
            out[i] += attnProj[i];
        }
        float[] ff = k.rmsNorm().normalize(out, preFeedforwardLayerNorm, eps);
        float[] down = k.rmsNorm().normalize(mlp.mlp(ff), postFeedforwardLayerNorm, eps);
        for (int i = 0; i < hidden; i++) {
            out[i] += down[i];
        }
        return out;
    }

    /**
     * GPU-resident single-token decode step (GEMMA-WARP-13b-3a): same math as {@link #decodeStep} but the
     * whole layer runs on GPU-resident buffers — the only CPU readback is the new token's k/v into the
     * host KV cache (2 per layer) instead of one readback per kernel. Returns the layer output as a
     * resident buffer (the caller closes the input buffer it passed in).
     */
    public WarpGpuBuffer decodeStepResident(WarpExecutionContext ctx, WarpGpuBuffer hiddenIn, int pos,
                                            Gemma3WarpKvCache cache) throws WindowsNativeException {
        ensureOpen();
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(hiddenIn, "hiddenIn");
        Objects.requireNonNull(cache, "cache");
        ensureResidentNormWeights(ctx);

        WarpGpuBuffer normed = k.rmsNorm().normalize(ctx, hiddenIn, inputLayerNormBuf, eps);
        WarpGpuBuffer q = qProj.forwardResident(ctx, normed);
        WarpGpuBuffer k0 = kProj.forwardResident(ctx, normed);
        WarpGpuBuffer v = vProj.forwardResident(ctx, normed);
        normed.close();
        WarpGpuBuffer qn = k.qkNorm().normalizeHeads(ctx, q, numHeads, headDim, qNormBuf, eps);
        q.close();
        WarpGpuBuffer kn = k.qkNorm().normalizeHeads(ctx, k0, numKvHeads, headDim, kNormBuf, eps);
        k0.close();
        WarpGpuBuffer qr = k.rope().applyToHeads(ctx, qn, numHeads, headDim, pos, theta);
        qn.close();
        WarpGpuBuffer kr = k.rope().applyToHeads(ctx, kn, numKvHeads, headDim, pos, theta);
        kn.close();

        // Only readback: the new token's k (normed+roped) and v (raw) into the host KV cache.
        cache.put(layer, pos, kr.readback(), v.readback());
        kr.close();
        v.close();
        int seqLen = pos + 1;
        int firstValid = layout.firstValidKey(layer, pos);
        WarpGpuBuffer keys = ctx.upload(cache.kFlat(layer, seqLen));
        WarpGpuBuffer values = ctx.upload(cache.vFlat(layer, seqLen));

        WarpGpuBuffer scores = k.scores().scores(ctx, qr, keys, numHeads, numKvHeads, headDim, seqLen, pos, firstValid, scale);
        qr.close();
        keys.close();
        WarpGpuBuffer prob = k.softmax().softmaxRows(ctx, scores, numHeads, seqLen);
        scores.close();
        WarpGpuBuffer context = k.value().aggregate(ctx, prob, values, numHeads, numKvHeads, headDim, seqLen);
        prob.close();
        values.close();
        WarpGpuBuffer attnProj = oProj.forwardResident(ctx, context);
        context.close();
        WarpGpuBuffer attnProjNorm = k.rmsNorm().normalize(ctx, attnProj, postAttentionLayerNormBuf, eps);
        attnProj.close();
        WarpGpuBuffer hidden1 = k.elementAdd().add(ctx, hiddenIn, attnProjNorm);
        attnProjNorm.close();

        WarpGpuBuffer ff = k.rmsNorm().normalize(ctx, hidden1, preFeedforwardLayerNormBuf, eps);
        WarpGpuBuffer down = mlp.mlp(ctx, ff);
        ff.close();
        WarpGpuBuffer downNorm = k.rmsNorm().normalize(ctx, down, postFeedforwardLayerNormBuf, eps);
        down.close();
        WarpGpuBuffer out = k.elementAdd().add(ctx, hidden1, downNorm);
        hidden1.close();
        downNorm.close();
        return out;
    }

    private void ensureResidentNormWeights(WarpExecutionContext ctx) throws WindowsNativeException {
        if (inputLayerNormBuf == null) {
            inputLayerNormBuf = ctx.upload(inputLayerNorm);
            qNormBuf = ctx.upload(qNorm);
            kNormBuf = ctx.upload(kNorm);
            postAttentionLayerNormBuf = ctx.upload(postAttentionLayerNorm);
            preFeedforwardLayerNormBuf = ctx.upload(preFeedforwardLayerNorm);
            postFeedforwardLayerNormBuf = ctx.upload(postFeedforwardLayerNorm);
        }
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
            closeBuffer(inputLayerNormBuf);
            closeBuffer(qNormBuf);
            closeBuffer(kNormBuf);
            closeBuffer(postAttentionLayerNormBuf);
            closeBuffer(preFeedforwardLayerNormBuf);
            closeBuffer(postFeedforwardLayerNormBuf);
            if (ownsKernels) {
                k.close();
            }
        }
    }

    private static void closeBuffer(WarpGpuBuffer b) {
        if (b != null) {
            b.close();
        }
    }
}
