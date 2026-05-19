package com.aresstack.windirectml.encoder.bert;

import com.aresstack.windirectml.encoder.PoolingStrategy;

import java.util.Objects;

/**
 * Generic configuration record for a BERT-style transformer encoder
 * (MiniLM, E5, JinaBERT, …).
 * <p>
 * Holds the architectural knobs that the shared DirectML encoder
 * pipeline ({@link DirectMlBertEncoderStack},
 * {@link DirectMlBertEncoderLayerBlock}) consumes – no MiniLM-specific
 * defaults or field names leak through here. Model families provide
 * their own factory (e.g. {@code BertEncoderConfig.minilm()}) and load
 * the remaining details (vocab, tokenizer) outside this record.
 * <p>
 * All fields are immutable. {@code headDim() = hiddenSize / numHeads}
 * is a derived view; callers must ensure the division is exact – this
 * is validated in {@link #validate()}.
 *
 * @param modelName             human-readable identifier (e.g. {@code "sentence-transformers/all-MiniLM-L6-v2"}).
 * @param hiddenSize            transformer hidden dimension {@code H}.
 * @param numLayers             number of encoder blocks stacked.
 * @param numHeads              attention heads per block ({@code H} must be a multiple).
 * @param intermediateSize      MLP feed-forward dimension {@code I}.
 * @param maxPositionEmbeddings maximum sequence length the position table supports.
 * @param typeVocabSize         token-type embedding rows (BERT-style: 2; some E5 variants: 1).
 * @param vocabSize             word-embedding vocab size.
 * @param layerNormEps          LayerNorm epsilon used inside the stack.
 * @param hiddenAct             activation in the MLP ({@code "gelu"}, …).
 * @param outputDimension       sentence-vector dimension after pooling.
 * @param poolingStrategy       pooling collapse {@code [seq,H] → [H]}.
 * @param normalize             whether the final vector is L2-normalised by default.
 */
public record BertEncoderConfig(
        String modelName,
        int hiddenSize,
        int numLayers,
        int numHeads,
        int intermediateSize,
        int maxPositionEmbeddings,
        int typeVocabSize,
        int vocabSize,
        float layerNormEps,
        String hiddenAct,
        int outputDimension,
        PoolingStrategy poolingStrategy,
        boolean normalize
) {

    public BertEncoderConfig {
        Objects.requireNonNull(modelName, "modelName");
        Objects.requireNonNull(hiddenAct, "hiddenAct");
        Objects.requireNonNull(poolingStrategy, "poolingStrategy");
    }

    /** Derived: {@code hiddenSize / numHeads}. */
    public int headDim() {
        return hiddenSize / numHeads;
    }

    /**
     * Throws {@link IllegalArgumentException} for any structural value
     * the DirectML stack cannot honour (zero/negative sizes,
     * {@code H % numHeads != 0}, etc.). Idempotent.
     */
    public void validate() {
        if (hiddenSize <= 0 || numLayers <= 0 || numHeads <= 0
                || intermediateSize <= 0 || maxPositionEmbeddings <= 0
                || typeVocabSize <= 0 || vocabSize <= 0 || outputDimension <= 0) {
            throw new IllegalArgumentException(
                    "BertEncoderConfig: all size fields must be > 0 (" + this + ")");
        }
        if (hiddenSize % numHeads != 0) {
            throw new IllegalArgumentException(
                    "hiddenSize (" + hiddenSize + ") must be a multiple of numHeads ("
                            + numHeads + ")");
        }
        if (layerNormEps <= 0f) {
            throw new IllegalArgumentException(
                    "layerNormEps must be > 0, got " + layerNormEps);
        }
    }

    /**
     * Convenience factory for {@code sentence-transformers/all-MiniLM-L6-v2}.
     * Values match the upstream {@code config.json}.
     */
    public static BertEncoderConfig minilm() {
        return new BertEncoderConfig(
                /* modelName             */ "sentence-transformers/all-MiniLM-L6-v2",
                /* hiddenSize            */ 384,
                /* numLayers             */ 6,
                /* numHeads              */ 12,
                /* intermediateSize      */ 1536,
                /* maxPositionEmbeddings */ 512,
                /* typeVocabSize         */ 2,
                /* vocabSize             */ 30522,
                /* layerNormEps          */ 1e-12f,
                /* hiddenAct             */ "gelu",
                /* outputDimension       */ 384,
                /* poolingStrategy       */ PoolingStrategy.MEAN,
                /* normalize             */ true
        );
    }
}

