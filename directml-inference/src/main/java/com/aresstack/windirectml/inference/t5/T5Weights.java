package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.RuntimeTensor;

import java.io.IOException;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Payload-backed T5 weights resolved from normalized wdmlpack tensor roles.
 *
 * <p>This class is intentionally still execution-neutral. Later encoder and
 * decoder pipelines should depend on this role-oriented view instead of
 * parsing manifests or source-format tensor names.</p>
 */
public final class T5Weights {
    private final T5PackageMetadata metadata;
    private final Map<String, RuntimeTensor> tensorsByRole;

    private T5Weights(T5PackageMetadata metadata, Map<String, RuntimeTensor> tensorsByRole) {
        this.metadata = Objects.requireNonNull(metadata, "metadata");
        this.tensorsByRole = Map.copyOf(new LinkedHashMap<>(tensorsByRole));
    }

    public static T5Weights load(T5RuntimePackage runtimePackage) throws IOException {
        Objects.requireNonNull(runtimePackage, "runtimePackage");
        return new T5Weights(runtimePackage.metadata(), new T5WeightResolver(runtimePackage).resolve());
    }

    public T5PackageMetadata metadata() {
        return metadata;
    }

    public RuntimeTensor require(String role) {
        RuntimeTensor tensor = tensorsByRole.get(role);
        if (tensor == null) {
            throw new IllegalArgumentException("Missing T5 tensor role: " + role);
        }
        return tensor;
    }

    public RuntimeTensor optional(String role) {
        return tensorsByRole.get(role);
    }

    public Collection<String> roles() {
        return tensorsByRole.keySet();
    }

    public int tensorCount() {
        return tensorsByRole.size();
    }

    public long payloadBytes() {
        return tensorsByRole.values().stream()
                .distinct()
                .mapToLong(RuntimeTensor::rawByteLength)
                .sum();
    }

    public RuntimeTensor sharedEmbedding() {
        return require("shared_embedding");
    }

    public RuntimeTensor lmHead() {
        return require("lm_head");
    }

    public RuntimeTensor encoderFinalLayerNorm() {
        return require("encoder.final_layer_norm");
    }

    public RuntimeTensor decoderFinalLayerNorm() {
        return require("decoder.final_layer_norm");
    }

    public RuntimeTensor encoderRelativeAttentionBias() {
        return optional("encoder.relative_attention_bias");
    }

    public RuntimeTensor decoderRelativeAttentionBias() {
        return optional("decoder.relative_attention_bias");
    }

    public RuntimeTensor encoderSelfAttention(int layer, String projection) {
        return require(role("encoder", layer, "self_attn." + projection));
    }

    public RuntimeTensor decoderSelfAttention(int layer, String projection) {
        return require(role("decoder", layer, "self_attn." + projection));
    }

    public RuntimeTensor decoderCrossAttention(int layer, String projection) {
        return require(role("decoder", layer, "cross_attn." + projection));
    }

    public RuntimeTensor encoderLayerNorm(int layer, int subLayer) {
        return require(role("encoder", layer, "layer_norm." + subLayer));
    }

    public RuntimeTensor decoderLayerNorm(int layer, int subLayer) {
        return require(role("decoder", layer, "layer_norm." + subLayer));
    }

    public RuntimeTensor encoderFeedForward(int layer, String weight) {
        return require(role("encoder", layer, "ffn." + weight));
    }

    public RuntimeTensor decoderFeedForward(int layer, String weight) {
        return require(role("decoder", layer, "ffn." + weight));
    }

    private static String role(String stack, int layer, String part) {
        return stack + ".layer." + layer + "." + part;
    }
}
