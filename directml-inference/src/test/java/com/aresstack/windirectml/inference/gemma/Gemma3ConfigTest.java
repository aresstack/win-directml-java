package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** GEMMA-WARP-1: config parsing + Gemma-specific derived values, against the real 270M-it config. */
class Gemma3ConfigTest {

    // Faithful subset of the real google/gemma-3-270m-it config.json.
    private static final String REAL_CONFIG = """
            {
              "architectures": ["Gemma3ForCausalLM"],
              "model_type": "gemma3_text",
              "hidden_size": 640,
              "intermediate_size": 2048,
              "num_hidden_layers": 18,
              "num_attention_heads": 4,
              "num_key_value_heads": 1,
              "head_dim": 256,
              "vocab_size": 262144,
              "max_position_embeddings": 32768,
              "sliding_window": 512,
              "_sliding_window_pattern": 6,
              "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention",
                "sliding_attention","sliding_attention","sliding_attention","full_attention"],
              "rope_theta": 1000000.0,
              "rope_local_base_freq": 10000.0,
              "query_pre_attn_scalar": 256,
              "rms_norm_eps": 1e-06,
              "hidden_activation": "gelu_pytorch_tanh",
              "bos_token_id": 2,
              "eos_token_id": 1,
              "pad_token_id": 0
            }
            """;

    private static Gemma3Config parse() throws Exception {
        return new Gemma3ConfigReader().read(REAL_CONFIG);
    }

    @Test
    void parsesCoreDimensions() throws Exception {
        Gemma3Config c = parse();
        assertEquals(640, c.hiddenSize());
        assertEquals(2048, c.intermediateSize());
        assertEquals(18, c.numHiddenLayers());
        assertEquals(4, c.numAttentionHeads());
        assertEquals(1, c.numKeyValueHeads());
        assertEquals(256, c.headDim());
        assertEquals(262144, c.vocabSize());
        assertEquals(512, c.slidingWindow());
        assertEquals("gelu_pytorch_tanh", c.hiddenActivation());
        assertEquals(2, c.bosTokenId());
        assertEquals(1, c.eosTokenId());
        assertEquals(0, c.padTokenId());
        assertTrue(c.tieWordEmbeddings());
    }

    @Test
    void headDimIsDecoupledFromHidden() throws Exception {
        Gemma3Config c = parse();
        assertEquals(1024, c.attentionDim());     // 4 heads * 256 != hidden(640)
        assertEquals(256, c.keyValueDim());        // 1 kv head * 256
        assertEquals(1.0 / Math.sqrt(256), c.attentionScale(), 1e-12);
        assertEquals(Math.sqrt(640), c.embeddingScale(), 1e-9);
    }

    @Test
    void fullAttentionLayersFromLayerTypes() throws Exception {
        Gemma3Config c = parse();
        for (int i = 0; i < c.numHiddenLayers(); i++) {
            boolean expectFull = (i == 5 || i == 11 || i == 17);
            assertEquals(expectFull, c.isFullAttentionLayer(i), "layer " + i);
        }
    }

    @Test
    void dualRopeBaseFrequenciesPerLayer() throws Exception {
        Gemma3Config c = parse();
        assertEquals(1_000_000.0, c.ropeThetaForLayer(5), 1e-6);   // full -> global
        assertEquals(10_000.0, c.ropeThetaForLayer(0), 1e-6);      // sliding -> local
    }

    @Test
    void fallsBackToPatternWhenLayerTypesAbsent() throws Exception {
        String noLayerTypes = REAL_CONFIG.replaceAll("\"layer_types\".*?],", "");
        Gemma3Config c = new Gemma3ConfigReader().read(noLayerTypes);
        // pattern 6 -> full at (i+1)%6==0 -> layers 5, 11, 17
        assertTrue(c.isFullAttentionLayer(5));
        assertFalse(c.isFullAttentionLayer(4));
        assertTrue(c.isFullAttentionLayer(11));
    }
}
