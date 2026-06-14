package com.aresstack.windirectml.inference.gemma;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Canonical Hugging Face Gemma 3 (text) tensor names. Gemma adds, relative to Llama/Qwen, the QK-norm
 * ({@code q_norm}/{@code k_norm}) and the sandwich norms ({@code pre/post_feedforward_layernorm}); the
 * LM head is tied to {@code embed_tokens} (no {@code lm_head.weight}).
 */
public final class Gemma3TensorNameMapper {

    public static final String EMBED_TOKENS = "model.embed_tokens.weight";
    public static final String FINAL_NORM = "model.norm.weight";

    private Gemma3TensorNameMapper() {
    }

    public static String inputLayerNorm(int layer) {
        return "model.layers." + layer + ".input_layernorm.weight";
    }

    public static String qProj(int layer) {
        return "model.layers." + layer + ".self_attn.q_proj.weight";
    }

    public static String kProj(int layer) {
        return "model.layers." + layer + ".self_attn.k_proj.weight";
    }

    public static String vProj(int layer) {
        return "model.layers." + layer + ".self_attn.v_proj.weight";
    }

    public static String oProj(int layer) {
        return "model.layers." + layer + ".self_attn.o_proj.weight";
    }

    public static String qNorm(int layer) {
        return "model.layers." + layer + ".self_attn.q_norm.weight";
    }

    public static String kNorm(int layer) {
        return "model.layers." + layer + ".self_attn.k_norm.weight";
    }

    public static String postAttentionLayerNorm(int layer) {
        return "model.layers." + layer + ".post_attention_layernorm.weight";
    }

    public static String preFeedforwardLayerNorm(int layer) {
        return "model.layers." + layer + ".pre_feedforward_layernorm.weight";
    }

    public static String gateProj(int layer) {
        return "model.layers." + layer + ".mlp.gate_proj.weight";
    }

    public static String upProj(int layer) {
        return "model.layers." + layer + ".mlp.up_proj.weight";
    }

    public static String downProj(int layer) {
        return "model.layers." + layer + ".mlp.down_proj.weight";
    }

    public static String postFeedforwardLayerNorm(int layer) {
        return "model.layers." + layer + ".post_feedforward_layernorm.weight";
    }

    /** All required tensor names for a config, in load order (embeddings + per-layer + final norm). */
    public static List<String> requiredTensorNames(Gemma3Config config) {
        List<String> names = new ArrayList<>();
        names.add(EMBED_TOKENS);
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            names.add(inputLayerNorm(i));
            names.add(qProj(i));
            names.add(kProj(i));
            names.add(vProj(i));
            names.add(oProj(i));
            names.add(qNorm(i));
            names.add(kNorm(i));
            names.add(postAttentionLayerNorm(i));
            names.add(preFeedforwardLayerNorm(i));
            names.add(gateProj(i));
            names.add(upProj(i));
            names.add(downProj(i));
            names.add(postFeedforwardLayerNorm(i));
        }
        names.add(FINAL_NORM);
        return names;
    }

    /** Expected 2D/1D shapes for each required tensor (for shape validation in the compiler). */
    public static Map<String, long[]> expectedShapes(Gemma3Config config) {
        int h = config.hiddenSize();
        int attn = config.attentionDim();
        int kv = config.keyValueDim();
        int inter = config.intermediateSize();
        int headDim = config.headDim();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        shapes.put(EMBED_TOKENS, new long[]{config.vocabSize(), h});
        for (int i = 0; i < config.numHiddenLayers(); i++) {
            shapes.put(inputLayerNorm(i), new long[]{h});
            shapes.put(qProj(i), new long[]{attn, h});
            shapes.put(kProj(i), new long[]{kv, h});
            shapes.put(vProj(i), new long[]{kv, h});
            shapes.put(oProj(i), new long[]{h, attn});
            shapes.put(qNorm(i), new long[]{headDim});
            shapes.put(kNorm(i), new long[]{headDim});
            shapes.put(postAttentionLayerNorm(i), new long[]{h});
            shapes.put(preFeedforwardLayerNorm(i), new long[]{h});
            shapes.put(gateProj(i), new long[]{inter, h});
            shapes.put(upProj(i), new long[]{inter, h});
            shapes.put(downProj(i), new long[]{h, inter});
            shapes.put(postFeedforwardLayerNorm(i), new long[]{h});
        }
        shapes.put(FINAL_NORM, new long[]{h});
        return shapes;
    }
}
