package com.aresstack.windirectml.inference.smollm2;

/**
 * Semantic SmolLM2 tensor roles independent of Hugging Face file names.
 */
public enum SmolLM2TensorRole {
    TOKEN_EMBEDDING(false),
    LAYER_INPUT_NORM(true),
    LAYER_SELF_Q(true),
    LAYER_SELF_K(true),
    LAYER_SELF_V(true),
    LAYER_SELF_O(true),
    LAYER_POST_ATTENTION_NORM(true),
    LAYER_MLP_GATE(true),
    LAYER_MLP_UP(true),
    LAYER_MLP_DOWN(true),
    FINAL_NORM(false),
    LM_HEAD(false);

    private final boolean layerBound;

    SmolLM2TensorRole(boolean layerBound) {
        this.layerBound = layerBound;
    }

    public boolean layerBound() {
        return layerBound;
    }
}
