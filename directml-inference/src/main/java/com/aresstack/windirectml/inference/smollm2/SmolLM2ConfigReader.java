package com.aresstack.windirectml.inference.smollm2;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads the SmolLM2/Llama CausalLM config.json subset used by import tooling.
 */
public final class SmolLM2ConfigReader {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public SmolLM2Config read(Path configPath) throws IOException {
        if (configPath == null || !Files.isRegularFile(configPath)) {
            throw new IOException("missing config.json: " + configPath);
        }
        return read(Files.readString(configPath));
    }

    public SmolLM2Config read(String json) throws IOException {
        JsonNode root = MAPPER.readTree(json);
        int hiddenSize = requiredInt(root, "hidden_size");
        int attentionHeads = requiredInt(root, "num_attention_heads");
        return new SmolLM2Config(
                text(root, "model_type", ""),
                architectures(root.get("architectures")),
                hiddenSize,
                requiredInt(root, "intermediate_size"),
                requiredInt(root, "num_hidden_layers"),
                attentionHeads,
                optionalInt(root, "num_key_value_heads"),
                optionalInt(root, "head_dim"),
                requiredInt(root, "vocab_size"),
                intValue(root, "max_position_embeddings", 0),
                doubleValue(root, "rms_norm_eps", 1.0e-5d),
                doubleValue(root, "rope_theta", 10000.0d),
                text(root, "hidden_act", ""),
                booleanValue(root, "attention_bias", false),
                booleanValue(root, "mlp_bias", false),
                intValue(root, "bos_token_id", 1),
                intValue(root, "eos_token_id", 2),
                optionalInt(root, "pad_token_id"),
                booleanValue(root, "tie_word_embeddings", false));
    }

    private static List<String> architectures(JsonNode node) {
        if (node == null || !node.isArray()) {
            return List.of();
        }
        List<String> values = new ArrayList<>();
        for (JsonNode item : node) {
            values.add(item.asText());
        }
        return values;
    }

    private static int requiredInt(JsonNode root, String field) throws IOException {
        JsonNode value = root.get(field);
        if (value == null || !value.canConvertToInt()) {
            throw new IOException("missing numeric config field: " + field);
        }
        return value.asInt();
    }

    private static int intValue(JsonNode root, String field, int defaultValue) {
        JsonNode value = root.get(field);
        return value != null && value.canConvertToInt() ? value.asInt() : defaultValue;
    }

    private static Integer optionalInt(JsonNode root, String field) {
        JsonNode value = root.get(field);
        return value != null && !value.isNull() && value.canConvertToInt() ? value.asInt() : null;
    }

    private static double doubleValue(JsonNode root, String field, double defaultValue) {
        JsonNode value = root.get(field);
        return value != null && value.isNumber() ? value.asDouble() : defaultValue;
    }

    private static boolean booleanValue(JsonNode root, String field, boolean defaultValue) {
        JsonNode value = root.get(field);
        return value != null && value.isBoolean() ? value.asBoolean() : defaultValue;
    }

    private static String text(JsonNode root, String field, String defaultValue) {
        JsonNode value = root.get(field);
        return value != null && value.isTextual() ? value.asText() : defaultValue;
    }
}
