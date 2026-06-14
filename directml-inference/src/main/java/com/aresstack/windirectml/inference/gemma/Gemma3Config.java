package com.aresstack.windirectml.inference.gemma;

import java.util.List;

/**
 * Typed subset of the Hugging Face Gemma 3 (text) {@code config.json}.
 *
 * <p>Gemma 3 deviates mathematically from the Llama/Qwen decoder-only schema (zero-centered RMSNorm,
 * GeGLU/GELU-tanh, QK-norm, sandwich norms, local/global attention with a sliding window, dual RoPE
 * base frequencies, {@code head_dim} decoupled from {@code hidden/heads}), so it gets its own family
 * instead of being pressed into the shared {@code decoderonly} layer.</p>
 */
public record Gemma3Config(
        String modelType,
        List<String> architectures,
        int hiddenSize,
        int intermediateSize,
        int numHiddenLayers,
        int numAttentionHeads,
        int numKeyValueHeads,
        int headDim,
        int vocabSize,
        int maxPositionEmbeddings,
        double rmsNormEps,
        double ropeThetaGlobal,
        double ropeThetaLocal,
        int slidingWindow,
        int slidingWindowPattern,
        List<String> layerTypes,
        double queryPreAttnScalar,
        String hiddenActivation,
        int bosTokenId,
        int eosTokenId,
        int padTokenId,
        boolean tieWordEmbeddings
) {
    public static final String FULL_ATTENTION = "full_attention";
    public static final String SLIDING_ATTENTION = "sliding_attention";

    public Gemma3Config {
        modelType = modelType == null ? "" : modelType;
        architectures = architectures == null ? List.of() : List.copyOf(architectures);
        layerTypes = layerTypes == null ? List.of() : List.copyOf(layerTypes);
        hiddenActivation = hiddenActivation == null ? "" : hiddenActivation;
    }

    /** Total projected attention width (heads * head_dim), which differs from hidden_size for Gemma. */
    public int attentionDim() {
        return numAttentionHeads * headDim;
    }

    /** Key/value projected width (kv_heads * head_dim). */
    public int keyValueDim() {
        return numKeyValueHeads * headDim;
    }

    /** Attention logit scale: {@code 1/sqrt(query_pre_attn_scalar)} (Gemma uses the configured scalar). */
    public double attentionScale() {
        double scalar = queryPreAttnScalar > 0 ? queryPreAttnScalar : headDim;
        return 1.0 / Math.sqrt(scalar);
    }

    /** Embedding scaling factor applied to input embeddings: {@code sqrt(hidden_size)}. */
    public double embeddingScale() {
        return Math.sqrt(hiddenSize);
    }

    /**
     * Whether layer {@code i} uses full causal attention (vs. sliding-window local attention). Driven by
     * the explicit {@code layer_types} list when present, else by the {@code _sliding_window_pattern}
     * (every {@code pattern}-th layer is global, i.e. {@code (i+1) % pattern == 0}).
     */
    public boolean isFullAttentionLayer(int layerIndex) {
        if (layerIndex >= 0 && layerIndex < layerTypes.size()) {
            return FULL_ATTENTION.equalsIgnoreCase(layerTypes.get(layerIndex));
        }
        if (slidingWindowPattern > 0) {
            return (layerIndex + 1) % slidingWindowPattern == 0;
        }
        return true;
    }

    /** RoPE base frequency for layer {@code i}: global theta for full-attention layers, local for sliding. */
    public double ropeThetaForLayer(int layerIndex) {
        return isFullAttentionLayer(layerIndex) ? ropeThetaGlobal : ropeThetaLocal;
    }

    public Gemma3SpecialTokens specialTokens() {
        return new Gemma3SpecialTokens(bosTokenId, eosTokenId, padTokenId);
    }
}
