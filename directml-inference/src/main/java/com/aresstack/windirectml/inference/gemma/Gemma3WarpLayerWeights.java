package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.warp.WarpWeightSource;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * Per-layer Gemma 3 weights for the WARP path (GEMMA-WARP-8 / -13a), row-major HF layout
 * ({@code [out, in]} for the projections).
 *
 * <p>The seven projections (q/k/v/o, gate/up/down) are exposed as {@link WarpWeightSource}s so the layer
 * uploads them heap-light from a direct FP32 {@link ByteBuffer} when available (product path), or from a
 * host {@code float[]} fallback (reference/tests) — the decision lives in
 * {@link com.aresstack.windirectml.inference.warp.WarpDenseProjection#fromWeightSource}. The small norm
 * vectors stay {@code float[]} (no buffering benefit).</p>
 *
 * <ul>
 *   <li>norms ({@code [hidden]}): {@code inputLayerNorm}, {@code postAttentionLayerNorm},
 *       {@code preFeedforwardLayerNorm}, {@code postFeedforwardLayerNorm}; ({@code [head_dim]})
 *       {@code qNorm}, {@code kNorm}</li>
 *   <li>{@code qProj} {@code [numHeads*head_dim, hidden]}; {@code kProj}/{@code vProj}
 *       {@code [numKvHeads*head_dim, hidden]}; {@code oProj} {@code [hidden, numHeads*head_dim]}</li>
 *   <li>{@code gateProj}/{@code upProj} {@code [intermediate, hidden]}; {@code downProj}
 *       {@code [hidden, intermediate]}</li>
 * </ul>
 */
public final class Gemma3WarpLayerWeights {

    private final float[] inputLayerNorm;
    private final float[] qNorm;
    private final float[] kNorm;
    private final float[] postAttentionLayerNorm;
    private final float[] preFeedforwardLayerNorm;
    private final float[] postFeedforwardLayerNorm;

    // Per projection: exactly one of (array, buffer) is non-null.
    private final float[] qArr;
    private final float[] kArr;
    private final float[] vArr;
    private final float[] oArr;
    private final float[] gateArr;
    private final float[] upArr;
    private final float[] downArr;
    private final ByteBuffer qBuf;
    private final ByteBuffer kBuf;
    private final ByteBuffer vBuf;
    private final ByteBuffer oBuf;
    private final ByteBuffer gateBuf;
    private final ByteBuffer upBuf;
    private final ByteBuffer downBuf;

    /** {@code float[]} projection form (reference/tests). */
    public Gemma3WarpLayerWeights(
            float[] inputLayerNorm,
            float[] qProj, float[] kProj, float[] vProj, float[] oProj,
            float[] qNorm, float[] kNorm,
            float[] postAttentionLayerNorm,
            float[] preFeedforwardLayerNorm,
            float[] gateProj, float[] upProj, float[] downProj,
            float[] postFeedforwardLayerNorm) {
        this(inputLayerNorm, qNorm, kNorm, postAttentionLayerNorm, preFeedforwardLayerNorm, postFeedforwardLayerNorm,
                req(qProj, "qProj"), req(kProj, "kProj"), req(vProj, "vProj"), req(oProj, "oProj"),
                req(gateProj, "gateProj"), req(upProj, "upProj"), req(downProj, "downProj"),
                null, null, null, null, null, null, null);
    }

    private Gemma3WarpLayerWeights(
            float[] inputLayerNorm, float[] qNorm, float[] kNorm, float[] postAttentionLayerNorm,
            float[] preFeedforwardLayerNorm, float[] postFeedforwardLayerNorm,
            float[] qArr, float[] kArr, float[] vArr, float[] oArr, float[] gateArr, float[] upArr, float[] downArr,
            ByteBuffer qBuf, ByteBuffer kBuf, ByteBuffer vBuf, ByteBuffer oBuf,
            ByteBuffer gateBuf, ByteBuffer upBuf, ByteBuffer downBuf) {
        this.inputLayerNorm = Objects.requireNonNull(inputLayerNorm, "inputLayerNorm");
        this.qNorm = Objects.requireNonNull(qNorm, "qNorm");
        this.kNorm = Objects.requireNonNull(kNorm, "kNorm");
        this.postAttentionLayerNorm = Objects.requireNonNull(postAttentionLayerNorm, "postAttentionLayerNorm");
        this.preFeedforwardLayerNorm = Objects.requireNonNull(preFeedforwardLayerNorm, "preFeedforwardLayerNorm");
        this.postFeedforwardLayerNorm = Objects.requireNonNull(postFeedforwardLayerNorm, "postFeedforwardLayerNorm");
        this.qArr = qArr;
        this.kArr = kArr;
        this.vArr = vArr;
        this.oArr = oArr;
        this.gateArr = gateArr;
        this.upArr = upArr;
        this.downArr = downArr;
        this.qBuf = qBuf;
        this.kBuf = kBuf;
        this.vBuf = vBuf;
        this.oBuf = oBuf;
        this.gateBuf = gateBuf;
        this.upBuf = upBuf;
        this.downBuf = downBuf;
    }

    /** Heap-light projection form: the seven projections as direct little-endian FP32 {@link ByteBuffer}s. */
    public static Gemma3WarpLayerWeights ofByteBufferProjections(
            float[] inputLayerNorm,
            ByteBuffer qProj, ByteBuffer kProj, ByteBuffer vProj, ByteBuffer oProj,
            float[] qNorm, float[] kNorm,
            float[] postAttentionLayerNorm,
            float[] preFeedforwardLayerNorm,
            ByteBuffer gateProj, ByteBuffer upProj, ByteBuffer downProj,
            float[] postFeedforwardLayerNorm) {
        return new Gemma3WarpLayerWeights(inputLayerNorm, qNorm, kNorm, postAttentionLayerNorm,
                preFeedforwardLayerNorm, postFeedforwardLayerNorm,
                null, null, null, null, null, null, null,
                req(qProj, "qProj"), req(kProj, "kProj"), req(vProj, "vProj"), req(oProj, "oProj"),
                req(gateProj, "gateProj"), req(upProj, "upProj"), req(downProj, "downProj"));
    }

    public static Gemma3WarpLayerWeights from(Gemma3ReferenceWeights.Layer layer) {
        Objects.requireNonNull(layer, "layer");
        return new Gemma3WarpLayerWeights(
                layer.inputLayerNorm, layer.qProj, layer.kProj, layer.vProj, layer.oProj,
                layer.qNorm, layer.kNorm, layer.postAttentionLayerNorm, layer.preFeedforwardLayerNorm,
                layer.gateProj, layer.upProj, layer.downProj, layer.postFeedforwardLayerNorm);
    }

    public float[] inputLayerNorm() {
        return inputLayerNorm;
    }

    public float[] qNorm() {
        return qNorm;
    }

    public float[] kNorm() {
        return kNorm;
    }

    public float[] postAttentionLayerNorm() {
        return postAttentionLayerNorm;
    }

    public float[] preFeedforwardLayerNorm() {
        return preFeedforwardLayerNorm;
    }

    public float[] postFeedforwardLayerNorm() {
        return postFeedforwardLayerNorm;
    }

    /** Whether the projections are backed by direct ByteBuffers (heap-light) rather than {@code float[]}. */
    public boolean hasByteBufferProjections() {
        return qBuf != null;
    }

    public WarpWeightSource qSource(int outputRows, int inputColumns) {
        return source("gemma3.q_proj", outputRows, inputColumns, qBuf, qArr);
    }

    public WarpWeightSource kSource(int outputRows, int inputColumns) {
        return source("gemma3.k_proj", outputRows, inputColumns, kBuf, kArr);
    }

    public WarpWeightSource vSource(int outputRows, int inputColumns) {
        return source("gemma3.v_proj", outputRows, inputColumns, vBuf, vArr);
    }

    public WarpWeightSource oSource(int outputRows, int inputColumns) {
        return source("gemma3.o_proj", outputRows, inputColumns, oBuf, oArr);
    }

    public WarpWeightSource gateSource(int outputRows, int inputColumns) {
        return source("gemma3.gate_proj", outputRows, inputColumns, gateBuf, gateArr);
    }

    public WarpWeightSource upSource(int outputRows, int inputColumns) {
        return source("gemma3.up_proj", outputRows, inputColumns, upBuf, upArr);
    }

    public WarpWeightSource downSource(int outputRows, int inputColumns) {
        return source("gemma3.down_proj", outputRows, inputColumns, downBuf, downArr);
    }

    private static WarpWeightSource source(String name, int outputRows, int inputColumns,
                                           ByteBuffer buf, float[] arr) {
        // ByteBuffer wins when present; the float[] supplier is only invoked otherwise.
        return WarpWeightSource.of(name, outputRows, inputColumns, buf, () -> arr);
    }

    private static float[] req(float[] a, String name) {
        return Objects.requireNonNull(a, name);
    }

    private static ByteBuffer req(ByteBuffer b, String name) {
        return Objects.requireNonNull(b, name);
    }
}
