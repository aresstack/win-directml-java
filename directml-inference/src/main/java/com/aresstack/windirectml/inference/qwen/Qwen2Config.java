package com.aresstack.windirectml.inference.qwen;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Model configuration for Qwen2.5-Coder-Instruct.
 * Parsed from the HuggingFace {@code config.json}.
 *
 * <p>Key architecture parameters for the 0.5B variant:
 * <ul>
 *   <li>hidden_size = 896</li>
 *   <li>num_attention_heads = 14</li>
 *   <li>num_hidden_layers = 24</li>
 *   <li>num_key_value_heads = 2 (GQA with 7:1 ratio)</li>
 *   <li>vocab_size = 151936</li>
 *   <li>max_position_embeddings = 32768</li>
 *   <li>intermediate_size = 4864</li>
 *   <li>rms_norm_eps = 1e-6</li>
 *   <li>rope_theta = 1000000.0</li>
 * </ul>
 *
 * <p><b>Differences from Phi-3:</b>
 * <ul>
 *   <li>GQA: Qwen 0.5B uses 2 KV heads for 14 Q heads (7:1 ratio).
 *       Phi-3 uses 32 KV heads = 32 Q heads (no grouping).</li>
 *   <li>rope_theta: Qwen uses 1,000,000 (long-context RoPE);
 *       Phi-3 uses 10,000.</li>
 *   <li>rms_norm_eps: Qwen uses 1e-6; Phi-3 uses 1e-5.</li>
 *   <li>Vocabulary: Qwen uses ~152k tokens; Phi-3 uses ~32k.</li>
 * </ul>
 *
 * <p>This config is model-driven: all parameters come from the JSON file,
 * making it compatible with 0.5B, 1.5B, and 3B variants without code changes.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record Qwen2Config(
        @JsonProperty("hidden_size") int hiddenSize,
        @JsonProperty("num_attention_heads") int numAttentionHeads,
        @JsonProperty("num_hidden_layers") int numHiddenLayers,
        @JsonProperty("num_key_value_heads") int numKeyValueHeads,
        @JsonProperty("vocab_size") int vocabSize,
        @JsonProperty("max_position_embeddings") int maxPositionEmbeddings,
        @JsonProperty("intermediate_size") int intermediateSize,
        @JsonProperty("rms_norm_eps") float rmsNormEps,
        @JsonProperty("rope_theta") float ropeTheta,
        @JsonProperty("tie_word_embeddings") boolean tieWordEmbeddings
) {
    /** Derived: dimension per attention head = hidden_size / num_attention_heads. */
    public int headDim() {
        return hiddenSize / numAttentionHeads;
    }

    /** Derived: size of Q projection output = num_attention_heads * head_dim. */
    public int qSize() {
        return numAttentionHeads * headDim();
    }

    /** Derived: size of K/V projection output = num_key_value_heads * head_dim. */
    public int kvSize() {
        return numKeyValueHeads * headDim();
    }

    /** Load config from a {@code config.json} file. */
    public static Qwen2Config load(Path configJson) throws IOException {
        if (configJson == null || !configJson.toFile().exists()) {
            throw new IOException("Qwen config.json not found: " + configJson);
        }
        return new ObjectMapper().readValue(configJson.toFile(), Qwen2Config.class);
    }
}
