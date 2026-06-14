package com.aresstack.windirectml.inference.gemma;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads the Gemma 3 (text) {@code config.json} subset used by the native runtime/compiler.
 */
public final class Gemma3ConfigReader {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public Gemma3Config read(Path configPath) throws IOException {
        if (configPath == null || !Files.isRegularFile(configPath)) {
            throw new IOException("missing Gemma config.json: " + configPath);
        }
        return read(Files.readString(configPath));
    }

    public Gemma3Config read(String json) throws IOException {
        JsonNode root = MAPPER.readTree(json);
        int headDim = optionalInt(root, "head_dim",
                requiredInt(root, "hidden_size") / requiredInt(root, "num_attention_heads"));
        return new Gemma3Config(
                text(root, "model_type", ""),
                stringArray(root.get("architectures")),
                requiredInt(root, "hidden_size"),
                requiredInt(root, "intermediate_size"),
                requiredInt(root, "num_hidden_layers"),
                requiredInt(root, "num_attention_heads"),
                optionalInt(root, "num_key_value_heads", requiredInt(root, "num_attention_heads")),
                headDim,
                requiredInt(root, "vocab_size"),
                optionalInt(root, "max_position_embeddings", 0),
                doubleValue(root, "rms_norm_eps", 1.0e-6d),
                doubleValue(root, "rope_theta", 1_000_000.0d),
                doubleValue(root, "rope_local_base_freq", 10_000.0d),
                optionalInt(root, "sliding_window", 0),
                optionalInt(root, "_sliding_window_pattern", 0),
                stringArray(root.get("layer_types")),
                doubleValue(root, "query_pre_attn_scalar", headDim),
                text(root, "hidden_activation", text(root, "hidden_act", "")),
                optionalInt(root, "bos_token_id", 2),
                optionalInt(root, "eos_token_id", 1),
                optionalInt(root, "pad_token_id", 0),
                // Gemma 3 ties embeddings/lm_head; config.json typically omits the flag.
                booleanValue(root, "tie_word_embeddings", true));
    }

    private static List<String> stringArray(JsonNode node) {
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
            throw new IOException("missing numeric Gemma config field: " + field);
        }
        return value.asInt();
    }

    private static int optionalInt(JsonNode root, String field, int defaultValue) {
        JsonNode value = root.get(field);
        return value != null && !value.isNull() && value.canConvertToInt() ? value.asInt() : defaultValue;
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
