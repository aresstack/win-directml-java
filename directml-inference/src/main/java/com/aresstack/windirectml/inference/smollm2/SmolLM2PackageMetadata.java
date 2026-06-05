package com.aresstack.windirectml.inference.smollm2;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * SmolLM2 package metadata written into wdmlpack manifests.
 */
public record SmolLM2PackageMetadata(
        String modelFamily,
        String architecture,
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
        String hiddenAct,
        boolean tieWordEmbeddings,
        int bosTokenId,
        int eosTokenId,
        Integer padTokenId
) {
    public static SmolLM2PackageMetadata from(SmolLM2Config config) {
        SmolLM2Architecture architecture = SmolLM2Architecture.from(config);
        return new SmolLM2PackageMetadata(
                "smollm2",
                "llama-causal-decoder",
                architecture.hiddenSize(),
                architecture.intermediateSize(),
                architecture.layers(),
                architecture.attentionHeads(),
                architecture.keyValueHeads(),
                architecture.headDim(),
                architecture.vocabSize(),
                architecture.maxPositionEmbeddings(),
                architecture.ropeTheta(),
                architecture.rmsNormEps(),
                config.hiddenAct(),
                architecture.usesTiedWordEmbeddings(),
                config.bosTokenId(),
                config.eosTokenId(),
                config.padTokenId());
    }

    public Map<String, Object> toManifest() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("modelFamily", modelFamily);
        out.put("architecture", architecture);
        out.put("hiddenSize", hiddenSize);
        out.put("intermediateSize", intermediateSize);
        out.put("layers", layers);
        out.put("attentionHeads", attentionHeads);
        out.put("keyValueHeads", keyValueHeads);
        out.put("headDim", headDim);
        out.put("vocabSize", vocabSize);
        out.put("maxPositionEmbeddings", maxPositionEmbeddings);
        out.put("ropeTheta", ropeTheta);
        out.put("rmsNormEps", rmsNormEps);
        out.put("hiddenAct", hiddenAct);
        out.put("tieWordEmbeddings", tieWordEmbeddings);
        out.put("bosTokenId", bosTokenId);
        out.put("eosTokenId", eosTokenId);
        if (padTokenId != null) {
            out.put("padTokenId", padTokenId);
        }
        return out;
    }
}
