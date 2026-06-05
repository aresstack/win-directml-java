package com.aresstack.windirectml.inference.smollm2;

/**
 * Runtime-relevant SmolLM2 architecture facts derived from config.json.
 */
public record SmolLM2Architecture(
        String modelSize,
        int hiddenSize,
        int intermediateSize,
        int layers,
        int attentionHeads,
        int keyValueHeads,
        int headDim,
        int vocabSize,
        int maxPositionEmbeddings,
        double ropeTheta,
        double rmsNormEps,
        boolean usesGroupedQueryAttention,
        boolean usesTiedWordEmbeddings
) {
    public SmolLM2Architecture {
        validatePositive("hiddenSize", hiddenSize);
        validatePositive("intermediateSize", intermediateSize);
        validatePositive("layers", layers);
        validatePositive("attentionHeads", attentionHeads);
        validatePositive("keyValueHeads", keyValueHeads);
        validatePositive("headDim", headDim);
        validatePositive("vocabSize", vocabSize);
        if (attentionHeads % keyValueHeads != 0) {
            throw new IllegalArgumentException("attentionHeads must be divisible by keyValueHeads");
        }
    }

    public static SmolLM2Architecture from(SmolLM2Config config) {
        int headDim = config.effectiveHeadDim();
        int keyValueHeads = config.effectiveKeyValueHeads();
        return new SmolLM2Architecture(
                detectModelSize(config),
                config.hiddenSize(),
                config.intermediateSize(),
                config.numHiddenLayers(),
                config.numAttentionHeads(),
                keyValueHeads,
                headDim,
                config.vocabSize(),
                config.maxPositionEmbeddings(),
                config.ropeTheta(),
                config.rmsNormEps(),
                config.numAttentionHeads() > keyValueHeads,
                config.tieWordEmbeddings());
    }

    private static String detectModelSize(SmolLM2Config config) {
        if (config.hiddenSize() <= 576 && config.numHiddenLayers() <= 30) {
            return "135m";
        }
        if (config.hiddenSize() <= 1024) {
            return "360m";
        }
        return "unknown";
    }

    private static void validatePositive(String name, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be > 0");
        }
    }
}
