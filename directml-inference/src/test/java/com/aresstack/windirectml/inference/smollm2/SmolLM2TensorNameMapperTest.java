package com.aresstack.windirectml.inference.smollm2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SmolLM2TensorNameMapperTest {

    private final SmolLM2TensorNameMapper mapper = new SmolLM2TensorNameMapper();

    @Test
    void mapsTokenEmbedding() {
        SmolLM2TensorRoleBinding binding = mapper.map("model.embed_tokens.weight").orElseThrow();

        assertEquals(SmolLM2TensorRole.TOKEN_EMBEDDING, binding.role());
        assertEquals(SmolLM2TensorRoleBinding.NO_LAYER, binding.layerIndex());
    }

    @Test
    void mapsLayerAttentionProjections() {
        assertLayerRole("model.layers.3.self_attn.q_proj.weight", SmolLM2TensorRole.LAYER_SELF_Q);
        assertLayerRole("model.layers.3.self_attn.k_proj.weight", SmolLM2TensorRole.LAYER_SELF_K);
        assertLayerRole("model.layers.3.self_attn.v_proj.weight", SmolLM2TensorRole.LAYER_SELF_V);
        assertLayerRole("model.layers.3.self_attn.o_proj.weight", SmolLM2TensorRole.LAYER_SELF_O);
    }

    @Test
    void mapsLayerMlpProjections() {
        assertLayerRole("model.layers.2.mlp.gate_proj.weight", SmolLM2TensorRole.LAYER_MLP_GATE);
        assertLayerRole("model.layers.2.mlp.up_proj.weight", SmolLM2TensorRole.LAYER_MLP_UP);
        assertLayerRole("model.layers.2.mlp.down_proj.weight", SmolLM2TensorRole.LAYER_MLP_DOWN);
    }

    private void assertLayerRole(String name, SmolLM2TensorRole role) {
        SmolLM2TensorRoleBinding binding = mapper.map(name).orElseThrow();
        assertEquals(role, binding.role());
        assertTrue(binding.layerIndex() >= 0);
    }
}
