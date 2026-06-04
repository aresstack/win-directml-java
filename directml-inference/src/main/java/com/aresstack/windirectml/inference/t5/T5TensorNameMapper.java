package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Maps Hugging Face T5/CodeT5 tensor names to stable IronMind runtime roles.
 *
 * <p>Keep this mapping out of the runtime. The runtime should know only the
 * normalized runtime names written into the wdmlpack manifest.</p>
 */
final class T5TensorNameMapper {

    private T5TensorNameMapper() {
    }

    static ExpectedTensor sharedEmbedding(T5Config config) {
        return required("shared_embedding", "shared.weight", "shared.embedding.weight",
                config.vocabSize(), config.modelSize());
    }

    static ExpectedTensor encoderFinalLayerNorm(T5Config config) {
        return required("encoder.final_layer_norm", "encoder.final_layer_norm.weight",
                "encoder.final_layer_norm.weight", config.modelSize());
    }

    static ExpectedTensor decoderFinalLayerNorm(T5Config config) {
        return required("decoder.final_layer_norm", "decoder.final_layer_norm.weight",
                "decoder.final_layer_norm.weight", config.modelSize());
    }

    static ExpectedTensor encoderSelfAttention(int layer, String projection, T5Config config) {
        long rows = "o".equals(projection) ? config.modelSize() : config.attentionInnerSize();
        long cols = "o".equals(projection) ? config.attentionInnerSize() : config.modelSize();
        return required(role("encoder", layer, "self_attn." + projection),
                "encoder.block." + layer + ".layer.0.SelfAttention." + projection + ".weight",
                runtimeName("encoder", layer, "self_attn", projection), rows, cols);
    }

    static ExpectedTensor decoderSelfAttention(int layer, String projection, T5Config config) {
        long rows = "o".equals(projection) ? config.modelSize() : config.attentionInnerSize();
        long cols = "o".equals(projection) ? config.attentionInnerSize() : config.modelSize();
        return required(role("decoder", layer, "self_attn." + projection),
                "decoder.block." + layer + ".layer.0.SelfAttention." + projection + ".weight",
                runtimeName("decoder", layer, "self_attn", projection), rows, cols);
    }

    static ExpectedTensor decoderCrossAttention(int layer, String projection, T5Config config) {
        long rows = "o".equals(projection) ? config.modelSize() : config.attentionInnerSize();
        long cols = "o".equals(projection) ? config.attentionInnerSize() : config.modelSize();
        return required(role("decoder", layer, "cross_attn." + projection),
                "decoder.block." + layer + ".layer.1.EncDecAttention." + projection + ".weight",
                runtimeName("decoder", layer, "cross_attn", projection), rows, cols);
    }

    static ExpectedTensor encoderLayerNorm(int layer, int subLayer, T5Config config) {
        return required(role("encoder", layer, "layer_norm." + subLayer),
                "encoder.block." + layer + ".layer." + subLayer + ".layer_norm.weight",
                runtimeName("encoder", layer, "layer_norm", Integer.toString(subLayer)), config.modelSize());
    }

    static ExpectedTensor decoderLayerNorm(int layer, int subLayer, T5Config config) {
        return required(role("decoder", layer, "layer_norm." + subLayer),
                "decoder.block." + layer + ".layer." + subLayer + ".layer_norm.weight",
                runtimeName("decoder", layer, "layer_norm", Integer.toString(subLayer)), config.modelSize());
    }

    static ExpectedTensor encoderFeedForward(int layer, String weight, T5Config config) {
        long rows = "wo".equals(weight) ? config.modelSize() : config.feedForwardSize();
        long cols = "wo".equals(weight) ? config.feedForwardSize() : config.modelSize();
        return required(role("encoder", layer, "ffn." + weight),
                "encoder.block." + layer + ".layer.1.DenseReluDense." + weight + ".weight",
                runtimeName("encoder", layer, "ffn", weight), rows, cols);
    }

    static ExpectedTensor decoderFeedForward(int layer, String weight, T5Config config) {
        long rows = "wo".equals(weight) ? config.modelSize() : config.feedForwardSize();
        long cols = "wo".equals(weight) ? config.feedForwardSize() : config.modelSize();
        return required(role("decoder", layer, "ffn." + weight),
                "decoder.block." + layer + ".layer.2.DenseReluDense." + weight + ".weight",
                runtimeName("decoder", layer, "ffn", weight), rows, cols);
    }

    static ExpectedTensor encoderRelativeAttentionBias(T5Config config) {
        return optional("encoder.relative_attention_bias",
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                "encoder.relative_attention_bias.weight",
                config.relativeAttentionBuckets(), config.attentionHeads());
    }

    static ExpectedTensor decoderRelativeAttentionBias(T5Config config) {
        return optional("decoder.relative_attention_bias",
                "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                "decoder.relative_attention_bias.weight",
                config.relativeAttentionBuckets(), config.attentionHeads());
    }

    static ExpectedTensor lmHead(T5Config config) {
        return optional("lm_head", "lm_head.weight", "lm_head.weight",
                config.vocabSize(), config.modelSize());
    }

    static ExpectedTensor required(String role, String sourceName, String runtimeName, long... dims) {
        return new ExpectedTensor(role, sourceName, runtimeName, true, dims);
    }

    static ExpectedTensor optional(String role, String sourceName, String runtimeName, long... dims) {
        return new ExpectedTensor(role, sourceName, runtimeName, false, dims);
    }

    private static String role(String stack, int layer, String part) {
        return stack + ".layer." + layer + "." + part;
    }

    private static String runtimeName(String stack, int layer, String group, String part) {
        return stack + ".layers." + padded(layer) + "." + group + "." + part + ".weight";
    }

    private static String padded(int layer) {
        if (layer < 10) {
            return "00" + layer;
        }
        if (layer < 100) {
            return "0" + layer;
        }
        return Integer.toString(layer);
    }

    record ExpectedTensor(String role,
                          String sourceName,
                          String runtimeName,
                          boolean required,
                          long[] expectedDims) {
        ExpectedTensor {
            Objects.requireNonNull(role, "role");
            Objects.requireNonNull(sourceName, "sourceName");
            Objects.requireNonNull(runtimeName, "runtimeName");
            expectedDims = expectedDims == null ? new long[0] : expectedDims.clone();
        }

        @Override
        public long[] expectedDims() {
            return expectedDims.clone();
        }
    }
}
