package com.aresstack.windirectml.sidecar.protocol.validation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class ModelValidator {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ModelValidator() {
    }

    public static ModelExpectation minilmExpectation() {
        return ModelExpectation.builder("MiniLM all-MiniLM-L6-v2")
                .require("config.json", "tokenizer.json")
                .eitherOf("model.safetensors", "pytorch_model.bin")
                .shape(384, 6, 12)
                .tokenizerType("WordPiece")
                .build();
    }

    public static ModelExpectation e5Expectation(String variant) {
        String v = variant == null ? "base-v2" : variant.trim().toLowerCase(Locale.ROOT);
        if ("small".equals(v) || "small-v2".equals(v) || "e5-small-v2".equals(v)) {
            return e5Ready("E5 small-v2", 384, 12, 12);
        }
        if ("large".equals(v) || "large-v2".equals(v) || "e5-large-v2".equals(v)) {
            return e5Ready("E5 large-v2", 1024, 24, 16);
        }
        if ("base-sts-en-de".equals(v) || "e5-base-sts-en-de".equals(v) || "sts-en-de".equals(v)) {
            return ModelExpectation.builder("E5 base-sts-en-de")
                    .require("config.json", "tokenizer.json")
                    .eitherOf("model.safetensors", "pytorch_model.bin")
                    .shape(768, 12, 12)
                    .tokenizerType("SentencePiece")
                    .notReady("planned: current upstream checkpoint is XLM-R/SentencePiece; current runtime supports WordPiece E5 variants only")
                    .build();
        }
        return e5Ready("E5 base-v2", 768, 12, 12);
    }

    private static ModelExpectation e5Ready(String label, int hidden, int layers, int heads) {
        return ModelExpectation.builder(label)
                .require("config.json", "tokenizer.json")
                .eitherOf("model.safetensors", "pytorch_model.bin")
                .shape(hidden, layers, heads)
                .tokenizerType("WordPiece")
                .build();
    }

    public static ModelExpectation rerankerExpectation() {
        return ModelExpectation.builder("Reranker cross-encoder")
                .require("config.json", "tokenizer.json")
                .eitherOf("model.safetensors", "pytorch_model.bin")
                .tokenizerType("WordPiece")
                .build();
    }

    /**
     * Expectation for Qwen2.5-Coder models (ONNX INT4 AWQ block-128 layout).
     * <p>
     * Marked as not ready because the ONNX source and runtime are still TBD/research.
     *
     * @param variant model size variant: "0.5b", "1.5b", or "3b"
     */
    public static ModelExpectation qwenExpectation(String variant) {
        String v = variant == null ? "0.5b" : variant.trim().toLowerCase(Locale.ROOT);
        String label = "Qwen2.5-Coder-" + v.toUpperCase(Locale.ROOT) + "-Instruct";
        ModelExpectation.Builder builder = ModelExpectation.builder(label)
                .require("config.json", "tokenizer.json", "tokenizer_config.json", "model.onnx")
                .eitherOf("model.onnx_data", "model.onnx.data")
                .tokenizerType("BPE")
                .notReady("planned: ONNX source TBD/research; runtime not yet implemented");
        // Shape values for known variants (hidden_size, num_hidden_layers, num_attention_heads)
        if ("0.5b".equals(v)) {
            builder.shape(896, 24, 14);
        } else if ("1.5b".equals(v)) {
            builder.shape(1536, 28, 12);
        } else if ("3b".equals(v)) {
            builder.shape(2048, 36, 16);
        }
        return builder.build();
    }

    public static ValidationReport validate(File dir, ModelExpectation expectation) {
        if (expectation == null) throw new IllegalArgumentException("expectation must not be null");
        List<ValidationFinding> findings = new ArrayList<ValidationFinding>();
        if (!expectation.isReady()) {
            findings.add(ValidationFinding.error(expectation.getNotReadyReason()));
        }
        if (dir == null) {
            findings.add(ValidationFinding.error("model directory is not configured"));
            return new ValidationReport(expectation.getLabel(), null, findings);
        }
        if (!dir.exists()) {
            findings.add(ValidationFinding.error("model directory does not exist: " + dir.getPath()));
            return new ValidationReport(expectation.getLabel(), dir, findings);
        }
        if (!dir.isDirectory()) {
            findings.add(ValidationFinding.error("model path is not a directory: " + dir.getPath()));
            return new ValidationReport(expectation.getLabel(), dir, findings);
        }
        checkFiles(dir, expectation, findings);
        File config = new File(dir, "config.json");
        if (config.isFile()) checkConfig(config, expectation, findings);
        File tokenizer = new File(dir, "tokenizer.json");
        if (tokenizer.isFile() && expectation.getTokenizerType() != null) {
            checkTokenizer(tokenizer, expectation.getTokenizerType(), findings);
        }
        return new ValidationReport(expectation.getLabel(), dir, findings);
    }

    private static void checkFiles(File dir, ModelExpectation expectation, List<ValidationFinding> findings) {
        for (String file : expectation.getRequiredFiles()) {
            if (new File(dir, file).isFile()) findings.add(ValidationFinding.ok("file present: " + file));
            else findings.add(ValidationFinding.error("missing required file: " + file));
        }
        for (List<String> group : expectation.getEitherOfFiles()) {
            boolean found = false;
            for (String file : group) {
                if (new File(dir, file).isFile()) {
                    found = true;
                    break;
                }
            }
            if (found) findings.add(ValidationFinding.ok("one weight file present: " + group));
            else findings.add(ValidationFinding.error("missing weight file; expected one of " + group));
        }
    }

    private static void checkConfig(File file, ModelExpectation e, List<ValidationFinding> findings) {
        JsonNode root;
        try {
            root = MAPPER.readTree(file);
        } catch (IOException ex) {
            findings.add(ValidationFinding.error("cannot parse config.json: " + ex.getMessage()));
            return;
        }
        checkInt(root, "hidden_size", e.getHiddenSize(), findings);
        checkInt(root, "num_hidden_layers", e.getLayers(), findings);
        checkInt(root, "num_attention_heads", e.getHeads(), findings);
    }

    private static void checkInt(JsonNode root, String name, Integer expected, List<ValidationFinding> findings) {
        if (expected == null) return;
        JsonNode value = root.get(name);
        if (value == null || !value.isNumber()) {
            findings.add(ValidationFinding.error("config.json missing numeric field: " + name));
            return;
        }
        int actual = value.asInt();
        if (actual == expected.intValue()) findings.add(ValidationFinding.ok(name + " = " + actual));
        else findings.add(ValidationFinding.error(name + " = " + actual + ", expected " + expected));
    }

    private static void checkTokenizer(File file, String expected, List<ValidationFinding> findings) {
        JsonNode root;
        try {
            root = MAPPER.readTree(file);
        } catch (IOException ex) {
            findings.add(ValidationFinding.error("cannot parse tokenizer.json: " + ex.getMessage()));
            return;
        }
        JsonNode model = root.get("model");
        JsonNode type = model == null ? null : model.get("type");
        if (type == null || !type.isTextual()) {
            findings.add(ValidationFinding.warn("tokenizer.json model.type is missing"));
            return;
        }
        String actual = type.asText();
        if (actual.equalsIgnoreCase(expected)) findings.add(ValidationFinding.ok("tokenizer type = " + actual));
        else findings.add(ValidationFinding.error("tokenizer type = " + actual + ", expected " + expected));
        if ("BPE".equalsIgnoreCase(expected)) {
            checkBpeSpecialTokens(root, findings);
        }
    }

    private static void checkBpeSpecialTokens(JsonNode tokenizerRoot, List<ValidationFinding> findings) {
        JsonNode added = tokenizerRoot.get("added_tokens");
        if (added == null || !added.isArray()) {
            findings.add(ValidationFinding.error("tokenizer.json missing added_tokens special-token metadata"));
            return;
        }
        boolean hasSpecial = false;
        boolean hasKnownStop = false;
        for (JsonNode token : added) {
            JsonNode content = token.get("content");
            JsonNode special = token.get("special");
            if (content == null || !content.isTextual() || special == null || !special.asBoolean(false)) {
                continue;
            }
            hasSpecial = true;
            String value = content.asText();
            if ("<|im_end|>".equals(value) || "<|endoftext|>".equals(value)) {
                hasKnownStop = true;
            }
        }
        if (hasSpecial && hasKnownStop) {
            findings.add(ValidationFinding.ok("tokenizer.json contains BPE special-token metadata"));
        } else if (hasSpecial) {
            findings.add(ValidationFinding.warn("tokenizer.json contains special tokens but no known Qwen stop token"));
        } else {
            findings.add(ValidationFinding.error("tokenizer.json contains no tokens marked special=true"));
        }
    }
}
