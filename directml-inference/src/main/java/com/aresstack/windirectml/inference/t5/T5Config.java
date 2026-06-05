package com.aresstack.windirectml.inference.t5;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Model configuration for the curated T5/CodeT5 seq2seq family.
 *
 * <p>Keep this type focused on architecture parameters that the import/compiler
 * layer needs to validate foreign tensor layouts before writing a wdmlpack. Do
 * not put runtime execution concerns into this config.</p>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record T5Config(
        @JsonProperty("architectures") List<String> architectures,
        @JsonProperty("model_type") String modelType,
        @JsonProperty("is_encoder_decoder") boolean encoderDecoder,
        @JsonProperty("d_model") int modelSize,
        @JsonProperty("d_kv") int keyValueSize,
        @JsonProperty("d_ff") int feedForwardSize,
        @JsonProperty("num_layers") int encoderLayers,
        @JsonProperty("num_decoder_layers") int decoderLayers,
        @JsonProperty("num_heads") int attentionHeads,
        @JsonProperty("vocab_size") int vocabSize,
        @JsonProperty("relative_attention_num_buckets") int relativeAttentionBuckets,
        @JsonProperty("relative_attention_max_distance") int relativeAttentionMaxDistance,
        @JsonProperty("layer_norm_epsilon") float layerNormEpsilon,
        @JsonProperty("decoder_start_token_id") int decoderStartTokenId,
        @JsonProperty("eos_token_id") int eosTokenId,
        @JsonProperty("pad_token_id") int padTokenId,
        @JsonProperty("tie_word_embeddings") Boolean tieWordEmbeddings,
        @JsonProperty("feed_forward_proj") String feedForwardProjection
) {
    /**
     * Load a T5/CodeT5 config from a Hugging Face {@code config.json} file.
     */
    public static T5Config load(Path configJson) throws IOException {
        if (configJson == null || !configJson.toFile().exists()) {
            throw new IOException("T5 config.json not found: " + configJson);
        }
        T5Config config = new ObjectMapper().readValue(configJson.toFile(), T5Config.class);
        config.validate();
        return config;
    }

    /**
     * Validate the small subset of T5/CodeT5 configurations supported by the compiler skeleton.
     */
    public void validate() throws IOException {
        if (!encoderDecoder) {
            throw new IOException("T5 config must be encoder-decoder");
        }
        if (modelType != null && !modelType.isBlank() && !"t5".equals(modelType)) {
            throw new IOException("Unsupported T5 model_type: " + modelType);
        }
        if (modelSize <= 0 || keyValueSize <= 0 || feedForwardSize <= 0
                || encoderLayers <= 0 || effectiveDecoderLayers() <= 0
                || attentionHeads <= 0 || vocabSize <= 0) {
            throw new IOException("T5 config contains non-positive architecture values");
        }
        if (relativeAttentionBuckets < 0 || relativeAttentionMaxDistance < 0) {
            throw new IOException("T5 config contains negative relative attention values");
        }
        try {
            specialTokens().validate();
        } catch (IllegalArgumentException e) {
            throw new IOException(e.getMessage(), e);
        }
    }

    /**
     * Return the decoder layer count, falling back to encoder layers for configs that omit it.
     */
    public int effectiveDecoderLayers() {
        return decoderLayers > 0 ? decoderLayers : encoderLayers;
    }

    /**
     * Return the inner attention projection size used by T5 q/k/v/o weights.
     */
    public int attentionInnerSize() {
        return attentionHeads * keyValueSize;
    }

    /**
     * Return whether the LM head is expected to share the embedding matrix.
     */
    public boolean usesTiedWordEmbeddings() {
        return tieWordEmbeddings == null || tieWordEmbeddings;
    }

    /**
     * Return whether the feed-forward block uses gated wi_0/wi_1 tensors.
     */
    public boolean usesGatedFeedForward() {
        if (feedForwardProjection == null) {
            return false;
        }
        return feedForwardProjection.startsWith("gated-") || feedForwardProjection.contains("gated");
    }

    public String effectiveFeedForwardProjection() {
        return feedForwardProjection == null || feedForwardProjection.isBlank() ? "relu" : feedForwardProjection;
    }

    public T5Architecture architecture() {
        return T5Architecture.from(this);
    }

    public T5SpecialTokens specialTokens() {
        return T5SpecialTokens.from(this);
    }
}
