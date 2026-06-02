package com.aresstack.windirectml.inference.phi3;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Model configuration for Phi-3-mini-4k-instruct.
 * Parsed from the HuggingFace {@code config.json}.
 *
 * <p>Key architecture parameters:
 * <ul>
 *   <li>hidden_size = 3072</li>
 *   <li>num_attention_heads = 32</li>
 *   <li>num_hidden_layers = 32</li>
 *   <li>num_key_value_heads = 32 (no GQA grouping)</li>
 *   <li>vocab_size = 32064</li>
 *   <li>max_position_embeddings = 4096</li>
 *   <li>intermediate_size = 8192</li>
 *   <li>rms_norm_eps = 1e-5</li>
 *   <li>rope_theta = 10000.0</li>
 * </ul>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record Phi3Config(
        @JsonProperty("hidden_size") int hiddenSize,
        @JsonProperty("num_attention_heads") int numAttentionHeads,
        @JsonProperty("num_hidden_layers") int numHiddenLayers,
        @JsonProperty("num_key_value_heads") int numKeyValueHeads,
        @JsonProperty("vocab_size") int vocabSize,
        @JsonProperty("max_position_embeddings") int maxPositionEmbeddings,
        @JsonProperty("intermediate_size") int intermediateSize,
        @JsonProperty("rms_norm_eps") float rmsNormEps,
        @JsonProperty("rope_theta") float ropeTheta
) {
    /**
     * Derived: dimension per attention head = hidden_size / num_attention_heads.
     */
    public int headDim() {
        return hiddenSize / numAttentionHeads;
    }

    /**
     * Load config from a {@code config.json} file.
     */
    public static Phi3Config load(Path configJson) throws IOException {
        return new ObjectMapper().readValue(configJson.toFile(), Phi3Config.class);
    }
}
