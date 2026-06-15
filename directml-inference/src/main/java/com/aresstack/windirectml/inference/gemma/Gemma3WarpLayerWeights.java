package com.aresstack.windirectml.inference.gemma;

import java.util.Objects;

/**
 * Per-layer Gemma 3 weights for the WARP single-layer step (GEMMA-WARP-8), row-major HF layout
 * ({@code [out, in]} for the projections). A clean WARP-side holder; {@link #from} builds it from the
 * loaded {@link Gemma3ReferenceWeights.Layer} so the runtime and the parity test share one source.
 *
 * <ul>
 *   <li>{@code inputLayerNorm}, {@code postAttentionLayerNorm}, {@code preFeedforwardLayerNorm},
 *       {@code postFeedforwardLayerNorm}: {@code [hidden]}</li>
 *   <li>{@code qNorm}, {@code kNorm}: {@code [head_dim]}</li>
 *   <li>{@code qProj}: {@code [numHeads*head_dim, hidden]}; {@code kProj}/{@code vProj}:
 *       {@code [numKvHeads*head_dim, hidden]}; {@code oProj}: {@code [hidden, numHeads*head_dim]}</li>
 *   <li>{@code gateProj}/{@code upProj}: {@code [intermediate, hidden]}; {@code downProj}:
 *       {@code [hidden, intermediate]}</li>
 * </ul>
 */
public record Gemma3WarpLayerWeights(
        float[] inputLayerNorm,
        float[] qProj, float[] kProj, float[] vProj, float[] oProj,
        float[] qNorm, float[] kNorm,
        float[] postAttentionLayerNorm,
        float[] preFeedforwardLayerNorm,
        float[] gateProj, float[] upProj, float[] downProj,
        float[] postFeedforwardLayerNorm) {

    public Gemma3WarpLayerWeights {
        Objects.requireNonNull(inputLayerNorm, "inputLayerNorm");
        Objects.requireNonNull(qProj, "qProj");
        Objects.requireNonNull(kProj, "kProj");
        Objects.requireNonNull(vProj, "vProj");
        Objects.requireNonNull(oProj, "oProj");
        Objects.requireNonNull(qNorm, "qNorm");
        Objects.requireNonNull(kNorm, "kNorm");
        Objects.requireNonNull(postAttentionLayerNorm, "postAttentionLayerNorm");
        Objects.requireNonNull(preFeedforwardLayerNorm, "preFeedforwardLayerNorm");
        Objects.requireNonNull(gateProj, "gateProj");
        Objects.requireNonNull(upProj, "upProj");
        Objects.requireNonNull(downProj, "downProj");
        Objects.requireNonNull(postFeedforwardLayerNorm, "postFeedforwardLayerNorm");
    }

    public static Gemma3WarpLayerWeights from(Gemma3ReferenceWeights.Layer layer) {
        Objects.requireNonNull(layer, "layer");
        return new Gemma3WarpLayerWeights(
                layer.inputLayerNorm, layer.qProj, layer.kProj, layer.vProj, layer.oProj,
                layer.qNorm, layer.kNorm, layer.postAttentionLayerNorm, layer.preFeedforwardLayerNorm,
                layer.gateProj, layer.upProj, layer.downProj, layer.postFeedforwardLayerNorm);
    }
}
