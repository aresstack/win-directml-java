package com.aresstack.windirectml.inference.decoderonly;

import java.util.Objects;

/**
 * GPU-resident projections + CPU-resident norm vectors for a single decoder-only layer.
 *
 * <p>The q/k/v matrices are fused into one {@code qkv_proj} dispatch and gate/up into one {@code gate_up_proj}
 * dispatch, so a layer issues four projection dispatches (qkv, o, gate_up, down) instead of seven. The slice index
 * constants describe how the fused outputs are laid out.</p>
 *
 * <p>This is a family-neutral data holder consumed by {@link DecoderOnlyWarpForwardPass}. The projections are built by
 * the model family from its own weight tensors and handed in; this class owns their lifecycle.</p>
 */
public final class DecoderOnlyWarpLayer implements AutoCloseable {

    /** Slice indices inside the fused {@code qkv_proj} output. */
    public static final int QKV_QUERY = 0;
    public static final int QKV_KEY = 1;
    public static final int QKV_VALUE = 2;

    /** Slice indices inside the fused {@code gate_up_proj} output. */
    public static final int GATE_UP_GATE = 0;
    public static final int GATE_UP_UP = 1;

    final float[] inputNorm;
    final float[] postAttentionNorm;
    final DecoderOnlyWarpFusedDenseProjection qkvProjection;
    final DecoderOnlyWarpDenseProjection outputProjection;
    final DecoderOnlyWarpFusedDenseProjection gateUpProjection;
    final DecoderOnlyWarpDenseProjection downProjection;

    public DecoderOnlyWarpLayer(float[] inputNorm,
                                float[] postAttentionNorm,
                                DecoderOnlyWarpFusedDenseProjection qkvProjection,
                                DecoderOnlyWarpDenseProjection outputProjection,
                                DecoderOnlyWarpFusedDenseProjection gateUpProjection,
                                DecoderOnlyWarpDenseProjection downProjection) {
        this.inputNorm = Objects.requireNonNull(inputNorm, "inputNorm");
        this.postAttentionNorm = Objects.requireNonNull(postAttentionNorm, "postAttentionNorm");
        this.qkvProjection = Objects.requireNonNull(qkvProjection, "qkvProjection");
        this.outputProjection = Objects.requireNonNull(outputProjection, "outputProjection");
        this.gateUpProjection = Objects.requireNonNull(gateUpProjection, "gateUpProjection");
        this.downProjection = Objects.requireNonNull(downProjection, "downProjection");
    }

    @Override
    public void close() {
        qkvProjection.close();
        outputProjection.close();
        gateUpProjection.close();
        downProjection.close();
    }
}
