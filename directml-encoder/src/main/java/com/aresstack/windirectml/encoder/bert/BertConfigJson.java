package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Reads a HuggingFace-style {@code config.json} into a
 * {@link BertEncoderConfig}.
 * <p>
 * Maps the canonical BERT/RoBERTa fields ({@code hidden_size},
 * {@code num_hidden_layers}, {@code num_attention_heads},
 * {@code intermediate_size}, {@code max_position_embeddings},
 * {@code type_vocab_size}, {@code vocab_size}, {@code layer_norm_eps},
 * {@code hidden_act}). The model name is taken from
 * {@code _name_or_path} when present, otherwise from the caller.
 * <p>
 * Used by the family loaders ({@code E5Encoders}, future
 * {@code BgeEncoders}, …) to validate that a user-declared variant
 * config (e.g. {@code e5-base-v2}) actually matches the checkpoint
 * on disk – mismatches surface as a clear {@link EmbeddingException}
 * instead of silent garbage vectors.
 */
public final class BertConfigJson {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private BertConfigJson() {
    }

    /**
     * Read {@code modelDir/config.json} and produce a
     * {@link BertEncoderConfig}.
     *
     * @param modelDir         directory containing {@code config.json}.
     * @param fallbackName     used as {@code modelName} when
     *                         {@code _name_or_path} is missing.
     * @param defaultPooling   {@link PoolingStrategy} (the JSON does
     *                         not encode it; the sentence-transformer
     *                         convention is {@link PoolingStrategy#MEAN}).
     * @param defaultNormalize whether the sentence vectors are
     *                         L2-normalised by default.
     * @return the parsed {@link BertEncoderConfig}, already
     * {@link BertEncoderConfig#validate() validated}.
     */
    public static BertEncoderConfig read(Path modelDir,
                                         String fallbackName,
                                         PoolingStrategy defaultPooling,
                                         boolean defaultNormalize) throws EmbeddingException {
        Path configJson = modelDir.resolve("config.json");
        if (!Files.exists(configJson)) {
            throw new EmbeddingException("Missing config.json in model directory: " + modelDir);
        }
        try {
            JsonNode root = MAPPER.readTree(configJson.toFile());
            String name = root.hasNonNull("_name_or_path")
                    ? root.get("_name_or_path").asText()
                    : fallbackName;
            int hidden = requireInt(root, "hidden_size");
            int numLayers = requireInt(root, "num_hidden_layers");
            int numHeads = requireInt(root, "num_attention_heads");
            int inter = requireInt(root, "intermediate_size");
            int maxPos = requireInt(root, "max_position_embeddings");
            int typeVocab = root.hasNonNull("type_vocab_size") ? root.get("type_vocab_size").asInt() : 2;
            int vocab = requireInt(root, "vocab_size");
            float lnEps = root.hasNonNull("layer_norm_eps")
                    ? (float) root.get("layer_norm_eps").asDouble()
                    : 1e-12f;
            String hiddenAct = root.hasNonNull("hidden_act") ? root.get("hidden_act").asText() : "gelu";
            // Normalise common HF GELU spellings to the kernel's plain "gelu".
            if (hiddenAct.equalsIgnoreCase("gelu_new")
                    || hiddenAct.equalsIgnoreCase("gelu_pytorch_tanh")
                    || hiddenAct.equalsIgnoreCase("gelu_fast")) {
                hiddenAct = "gelu";
            }

            BertEncoderConfig cfg = new BertEncoderConfig(
                    name,
                    hidden, numLayers, numHeads, inter,
                    maxPos, typeVocab, vocab,
                    lnEps, hiddenAct, /* outputDimension */ hidden,
                    defaultPooling, defaultNormalize);
            cfg.validate();
            return cfg;
        } catch (IOException e) {
            throw new EmbeddingException("Failed to read " + configJson + ": " + e.getMessage(), e);
        } catch (IllegalArgumentException e) {
            throw new EmbeddingException(
                    "config.json in " + modelDir + " is unusable: " + e.getMessage(), e);
        }
    }

    private static int requireInt(JsonNode root, String name) {
        if (!root.hasNonNull(name)) {
            throw new IllegalArgumentException("missing field '" + name + "'");
        }
        return root.get(name).asInt();
    }

    /**
     * Compare the declared/preset variant config against an on-disk
     * {@code config.json}; throws {@link EmbeddingException} on
     * mismatch on the shape-relevant axes. {@code modelName} and
     * {@code maxPositionEmbeddings} are intentionally tolerant
     * (different fine-tunes ship the same architecture under
     * different names; max-pos can be tightened down).
     */
    public static void verifyMatches(BertEncoderConfig declared, BertEncoderConfig onDisk,
                                     Path modelDir) throws EmbeddingException {
        if (declared.hiddenSize() != onDisk.hiddenSize()
                || declared.numLayers() != onDisk.numLayers()
                || declared.numHeads() != onDisk.numHeads()
                || declared.intermediateSize() != onDisk.intermediateSize()
                || declared.vocabSize() != onDisk.vocabSize()
                || declared.typeVocabSize() != onDisk.typeVocabSize()) {
            throw new EmbeddingException(
                    "E5 config mismatch between requested variant and model on disk at "
                            + modelDir + ":\n"
                            + "  requested: " + summary(declared) + "\n"
                            + "  on disk:   " + summary(onDisk)
                            + "\nPick a matching -De5.model or update -De5.modelDir.");
        }
    }

    private static String summary(BertEncoderConfig c) {
        return c.modelName()
                + " [hidden=" + c.hiddenSize()
                + ", layers=" + c.numLayers()
                + ", heads=" + c.numHeads()
                + ", inter=" + c.intermediateSize()
                + ", vocab=" + c.vocabSize()
                + ", typeVocab=" + c.typeVocabSize() + "]";
    }
}

