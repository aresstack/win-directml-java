package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsEntry;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsException;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsReader;
import com.aresstack.windirectml.runtime.TensorDataType;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Generic CPU-resident BERT-style weight bundle: embeddings tables
 * ({@code word}, {@code position}, {@code tokenType}), embedding LN
 * parameters and per-layer {@link BertCpuLayerWeights}. Driven by the
 * model-agnostic {@link BertEncoderConfig} so the same loader handles
 * MiniLM, E5-v2, BGE, … and every other HuggingFace BERT-named
 * checkpoint.
 * <p>
 * Tensor names follow the canonical {@code embeddings.*} /
 * {@code encoder.layer.{i}.*} HuggingFace convention. Common owning
 * prefixes (HF raw, sentence-transformers, BertModel-wrapped) are
 * probed automatically via {@link #PREFIX_CANDIDATES}.
 */
public final class BertCpuEncoderWeights {

    /**
     * Order of prefix candidates probed when resolving tensor names.
     */
    public static final String[] PREFIX_CANDIDATES = {
            "",
            "0.auto_model.",
            "bert.",
            "roberta.",
            "model.",
            "0.auto_model.bert.",
            "0.auto_model.roberta."
    };

    private final BertEncoderConfig config;
    public final float[] wordEmbeddings;        // [vocab, H]
    public final float[] positionEmbeddings;    // [maxPos, H]
    public final float[] tokenTypeEmbeddings;   // [typeVocab, H]
    public final float[] embLnGamma;            // [H]
    public final float[] embLnBeta;             // [H]
    public final List<BertCpuLayerWeights> layers; // numLayers

    private BertCpuEncoderWeights(BertEncoderConfig cfg,
                                  float[] we, float[] pe, float[] tte,
                                  float[] elng, float[] elnb,
                                  List<BertCpuLayerWeights> layers) {
        this.config = Objects.requireNonNull(cfg);
        this.wordEmbeddings = we;
        this.positionEmbeddings = pe;
        this.tokenTypeEmbeddings = tte;
        this.embLnGamma = elng;
        this.embLnBeta = elnb;
        this.layers = List.copyOf(layers);
    }

    public BertEncoderConfig config() {
        return config;
    }

    /**
     * Test-only factory accepting synthetic weights.
     */
    public static BertCpuEncoderWeights forTesting(BertEncoderConfig cfg,
                                                   float[] we, float[] pe, float[] tte,
                                                   float[] elng, float[] elnb,
                                                   List<BertCpuLayerWeights> layers) {
        if (layers.size() != cfg.numLayers()) {
            throw new IllegalArgumentException("layers.size()=" + layers.size()
                    + " but cfg.numLayers()=" + cfg.numLayers());
        }
        return new BertCpuEncoderWeights(cfg, we, pe, tte, elng, elnb, layers);
    }

    /**
     * Load BERT-style weights from {@code modelDir/model.safetensors}.
     */
    public static BertCpuEncoderWeights load(Path modelDir, BertEncoderConfig cfg)
            throws EmbeddingException {
        Path safetensors = modelDir.resolve("model.safetensors");
        try (SafetensorsReader reader = SafetensorsReader.open(safetensors)) {
            return loadFromReader(reader, cfg);
        } catch (SafetensorsException e) {
            throw new EmbeddingException("Failed to load BERT weights from " + safetensors, e);
        }
    }

    public static BertCpuEncoderWeights loadFromReader(SafetensorsReader reader, BertEncoderConfig cfg)
            throws SafetensorsException, EmbeddingException {
        cfg.validate();
        String prefix = detectPrefix(reader, "embeddings.word_embeddings.weight");

        float[] we = readFloats(reader, prefix + "embeddings.word_embeddings.weight");
        float[] pe = readFloats(reader, prefix + "embeddings.position_embeddings.weight");
        float[] tte = readFloats(reader, prefix + "embeddings.token_type_embeddings.weight");
        float[] elng = readFloats(reader, prefix + "embeddings.LayerNorm.weight");
        float[] elnb = readFloats(reader, prefix + "embeddings.LayerNorm.bias");

        validateLength("word_embeddings", we, (long) cfg.vocabSize() * cfg.hiddenSize());
        validateLength("position_embeddings", pe,
                (long) cfg.maxPositionEmbeddings() * cfg.hiddenSize());
        validateLength("token_type_embeddings", tte,
                (long) cfg.typeVocabSize() * cfg.hiddenSize());

        List<BertCpuLayerWeights> layers = new ArrayList<>(cfg.numLayers());
        for (int i = 0; i < cfg.numLayers(); i++) {
            String base = prefix + "encoder.layer." + i + ".";
            layers.add(new BertCpuLayerWeights(
                    readFloats(reader, base + "attention.self.query.weight"),
                    readFloats(reader, base + "attention.self.query.bias"),
                    readFloats(reader, base + "attention.self.key.weight"),
                    readFloats(reader, base + "attention.self.key.bias"),
                    readFloats(reader, base + "attention.self.value.weight"),
                    readFloats(reader, base + "attention.self.value.bias"),
                    readFloats(reader, base + "attention.output.dense.weight"),
                    readFloats(reader, base + "attention.output.dense.bias"),
                    readFloats(reader, base + "attention.output.LayerNorm.weight"),
                    readFloats(reader, base + "attention.output.LayerNorm.bias"),
                    readFloats(reader, base + "intermediate.dense.weight"),
                    readFloats(reader, base + "intermediate.dense.bias"),
                    readFloats(reader, base + "output.dense.weight"),
                    readFloats(reader, base + "output.dense.bias"),
                    readFloats(reader, base + "output.LayerNorm.weight"),
                    readFloats(reader, base + "output.LayerNorm.bias")));
        }
        return new BertCpuEncoderWeights(cfg, we, pe, tte, elng, elnb, layers);
    }

    private static String detectPrefix(SafetensorsReader reader, String suffix)
            throws EmbeddingException {
        for (String candidate : PREFIX_CANDIDATES) {
            if (reader.tensorNames().contains(candidate + suffix)) return candidate;
        }
        throw new EmbeddingException("Could not locate '" + suffix
                + "' under any known prefix " + Arrays.toString(PREFIX_CANDIDATES)
                + ". Available tensors (sample): "
                + reader.tensorNames().stream().limit(5).toList());
    }

    private static float[] readFloats(SafetensorsReader reader, String name)
            throws SafetensorsException {
        SafetensorsEntry entry = reader.entry(name);
        if (entry.dataType() != TensorDataType.FLOAT32) {
            throw new SafetensorsException("expected F32 for '" + name + "' but got " + entry.dataType());
        }
        return reader.readFloat32(name);
    }

    private static void validateLength(String name, float[] data, long expected)
            throws EmbeddingException {
        if (data.length != expected) {
            throw new EmbeddingException("tensor '" + name + "' length " + data.length
                    + " != expected " + expected);
        }
    }
}

