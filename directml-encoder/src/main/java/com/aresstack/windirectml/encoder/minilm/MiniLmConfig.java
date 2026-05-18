package com.aresstack.windirectml.encoder.minilm;

/**
 * Konkrete, unveränderliche Modellkonfiguration für
 * {@code sentence-transformers/all-MiniLM-L6-v2}.
 * <p>
 * Werte stammen aus dem offiziellen {@code config.json} des Modells.
 * Werden zur Validierung verwendet, wenn echte Gewichte geladen sind.
 */
public record MiniLmConfig(
        int hiddenSize,
        int numLayers,
        int numAttentionHeads,
        int intermediateSize,
        int maxPositionEmbeddings,
        int typeVocabSize,
        int vocabSize,
        float layerNormEps,
        String hiddenAct,
        int outputDimension
) {

    public static MiniLmConfig defaults() {
        return new MiniLmConfig(
                /* hiddenSize            */ 384,
                /* numLayers             */ 6,
                /* numAttentionHeads     */ 12,
                /* intermediateSize      */ 1536,
                /* maxPositionEmbeddings */ 512,
                /* typeVocabSize         */ 2,
                /* vocabSize             */ 30522,
                /* layerNormEps          */ 1e-12f,
                /* hiddenAct             */ "gelu",
                /* outputDimension       */ 384
        );
    }

    public int headDim() {
        return hiddenSize / numAttentionHeads;
    }
}

