package com.aresstack.windirectml.encoder.reranker;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.bert.BertCpuEncoderWeights;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsEntry;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsException;
import com.aresstack.windirectml.encoder.safetensors.SafetensorsReader;
import com.aresstack.windirectml.runtime.TensorDataType;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Cross-encoder weight bundle: the regular BERT encoder weights
 * ({@link BertCpuEncoderWeights}) plus the classification head
 * ({@code classifier.weight} of shape {@code [numLabels, H]} and
 * {@code classifier.bias} of shape {@code [numLabels]}).
 * <p>
 * For ranking models the convention is {@code numLabels = 1} – the
 * single logit is the relevance score and downstream code never
 * applies a softmax (cross-encoder logits are not calibrated across
 * models). Models with {@code numLabels > 1} (e.g. bge-reranker-v2-m3
 * with a 2-way classifier) are rejected for now; we will add explicit
 * support once the first such checkpoint lands.
 * <p>
 * HuggingFace cross-encoders sometimes also carry a pre-classifier
 * pooler ({@code bert.pooler.dense.*}), but {@code ms-marco-MiniLM-L-6-v2}
 * and similar SentenceTransformer-exported rerankers feed the raw
 * {@code [CLS]} hidden directly into the linear head, which is what we
 * implement here. The optional pooler is therefore intentionally
 * ignored: enabling it would silently change the score for models that
 * do not use it.
 */
public final class RerankerCpuWeights {

    /**
     * Shape: {@code [numLabels, H]}, row-major. For reranking {@code numLabels = 1}.
     */
    public final float[] classifierWeight;
    /**
     * Shape: {@code [numLabels]}.
     */
    public final float[] classifierBias;
    /**
     * Always {@code 1} for the supported reranker checkpoints.
     */
    public final int numLabels;
    private final BertCpuEncoderWeights bert;

    public RerankerCpuWeights(BertCpuEncoderWeights bert,
                              float[] classifierWeight,
                              float[] classifierBias,
                              int numLabels) {
        this.bert = Objects.requireNonNull(bert);
        this.classifierWeight = Objects.requireNonNull(classifierWeight);
        this.classifierBias = Objects.requireNonNull(classifierBias);
        this.numLabels = numLabels;
        int H = bert.config().hiddenSize();
        if (numLabels != 1) {
            throw new IllegalArgumentException(
                    "RerankerCpuWeights currently only supports numLabels=1, got " + numLabels);
        }
        if (classifierWeight.length != (long) numLabels * H) {
            throw new IllegalArgumentException("classifier.weight length " + classifierWeight.length
                    + " != " + numLabels + "*" + H);
        }
        if (classifierBias.length != numLabels) {
            throw new IllegalArgumentException("classifier.bias length " + classifierBias.length
                    + " != " + numLabels);
        }
    }

    public BertCpuEncoderWeights bert() {
        return bert;
    }

    public BertEncoderConfig config() {
        return bert.config();
    }

    /**
     * Load encoder + classifier-head weights from
     * {@code modelDir/model.safetensors}. The classifier head is
     * discovered under one of the common HuggingFace prefixes (no
     * prefix, {@code 0.}, …) – HF's {@code AutoModelForSequenceClassification}
     * places it next to {@code bert.*} or {@code roberta.*}, never
     * underneath them.
     */
    public static RerankerCpuWeights load(Path modelDir, BertEncoderConfig cfg)
            throws EmbeddingException {
        Path safetensors = modelDir.resolve("model.safetensors");
        try (SafetensorsReader reader = SafetensorsReader.open(safetensors)) {
            BertCpuEncoderWeights bert = BertCpuEncoderWeights.loadFromReader(reader, cfg);
            String classifierName = findClassifierTensor(reader);
            String classifierPrefix = classifierName.substring(
                    0, classifierName.length() - "classifier.weight".length());
            float[] w = readFloats(reader, classifierName);
            float[] b = readFloats(reader, classifierPrefix + "classifier.bias");

            int H = cfg.hiddenSize();
            int numLabels = w.length / H;
            if (numLabels * H != w.length) {
                throw new EmbeddingException("classifier.weight (" + w.length
                        + ") is not a multiple of H=" + H);
            }
            return new RerankerCpuWeights(bert, w, b, numLabels);
        } catch (SafetensorsException e) {
            throw new EmbeddingException("Failed to load reranker weights from " + safetensors, e);
        }
    }

    private static String findClassifierTensor(SafetensorsReader reader) throws EmbeddingException {
        for (String prefix : BertCpuEncoderWeights.PREFIX_CANDIDATES) {
            String name = prefix + "classifier.weight";
            if (reader.tensorNames().contains(name)) return name;
        }
        throw new EmbeddingException(
                "Could not locate a 'classifier.weight' tensor under any known prefix "
                        + java.util.Arrays.toString(BertCpuEncoderWeights.PREFIX_CANDIDATES)
                        + ". Is this a sentence-classification checkpoint?");
    }

    private static float[] readFloats(SafetensorsReader reader, String name)
            throws SafetensorsException {
        SafetensorsEntry entry = reader.entry(name);
        if (entry.dataType() != TensorDataType.FLOAT32) {
            throw new SafetensorsException("expected F32 for '" + name + "' but got " + entry.dataType());
        }
        return reader.readFloat32(name);
    }
}

