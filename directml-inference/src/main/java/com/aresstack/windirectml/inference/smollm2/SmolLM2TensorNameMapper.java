package com.aresstack.windirectml.inference.smollm2;

import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Maps Hugging Face Llama-style SmolLM2 tensor names to semantic roles.
 */
public final class SmolLM2TensorNameMapper {

    private static final Pattern LAYER_PATTERN = Pattern.compile("model\\.layers\\.(\\d+)\\.(.+)");

    public Optional<SmolLM2TensorRoleBinding> map(String tensorName) {
        if (tensorName == null || tensorName.isBlank()) {
            return Optional.empty();
        }
        SmolLM2TensorRole rootRole = rootRole(tensorName);
        if (rootRole != null) {
            return Optional.of(new SmolLM2TensorRoleBinding(rootRole, SmolLM2TensorRoleBinding.NO_LAYER, tensorName));
        }
        Matcher matcher = LAYER_PATTERN.matcher(tensorName);
        if (!matcher.matches()) {
            return Optional.empty();
        }
        int layerIndex = Integer.parseInt(matcher.group(1));
        SmolLM2TensorRole layerRole = layerRole(matcher.group(2));
        if (layerRole == null) {
            return Optional.empty();
        }
        return Optional.of(new SmolLM2TensorRoleBinding(layerRole, layerIndex, tensorName));
    }

    private static SmolLM2TensorRole rootRole(String tensorName) {
        return switch (tensorName) {
            case "model.embed_tokens.weight" -> SmolLM2TensorRole.TOKEN_EMBEDDING;
            case "model.norm.weight" -> SmolLM2TensorRole.FINAL_NORM;
            case "lm_head.weight" -> SmolLM2TensorRole.LM_HEAD;
            default -> null;
        };
    }

    private static SmolLM2TensorRole layerRole(String suffix) {
        return switch (suffix) {
            case "input_layernorm.weight" -> SmolLM2TensorRole.LAYER_INPUT_NORM;
            case "self_attn.q_proj.weight" -> SmolLM2TensorRole.LAYER_SELF_Q;
            case "self_attn.k_proj.weight" -> SmolLM2TensorRole.LAYER_SELF_K;
            case "self_attn.v_proj.weight" -> SmolLM2TensorRole.LAYER_SELF_V;
            case "self_attn.o_proj.weight" -> SmolLM2TensorRole.LAYER_SELF_O;
            case "post_attention_layernorm.weight" -> SmolLM2TensorRole.LAYER_POST_ATTENTION_NORM;
            case "mlp.gate_proj.weight" -> SmolLM2TensorRole.LAYER_MLP_GATE;
            case "mlp.up_proj.weight" -> SmolLM2TensorRole.LAYER_MLP_UP;
            case "mlp.down_proj.weight" -> SmolLM2TensorRole.LAYER_MLP_DOWN;
            default -> null;
        };
    }
}
