package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;

import java.util.LinkedHashMap;
import java.util.List;
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


    public static SmolLM2PackageMetadata fromManifest(Map<String, Object> manifest) {
        Map<String, Object> model = RuntimeModelPackage.castMap((Map<?, ?>) manifest.get("model"));
        return new SmolLM2PackageMetadata(
                RuntimeModelPackage.stringValue(manifest.get("modelFamily")),
                RuntimeModelPackage.stringValue(manifest.get("architecture")),
                RuntimeModelPackage.intValue(model.get("hiddenSize"), 0),
                RuntimeModelPackage.intValue(model.get("intermediateSize"), 0),
                RuntimeModelPackage.intValue(model.get("layers"), 0),
                RuntimeModelPackage.intValue(model.get("attentionHeads"), 0),
                RuntimeModelPackage.intValue(model.get("keyValueHeads"), 0),
                RuntimeModelPackage.intValue(model.get("headDim"), 0),
                RuntimeModelPackage.intValue(model.get("vocabSize"), 0),
                RuntimeModelPackage.intValue(model.get("maxPositionEmbeddings"), 0),
                doubleValue(model.get("ropeTheta"), 10000.0d),
                doubleValue(model.get("rmsNormEps"), 1.0e-5d),
                RuntimeModelPackage.stringValue(model.get("hiddenAct")),
                booleanValue(model.get("tieWordEmbeddings"), false),
                RuntimeModelPackage.intValue(model.get("bosTokenId"), 1),
                RuntimeModelPackage.intValue(model.get("eosTokenId"), 2),
                model.containsKey("padTokenId") ? RuntimeModelPackage.intValue(model.get("padTokenId"), 0) : null);
    }

    public SmolLM2Config toConfig() {
        return new SmolLM2Config(
                "llama",
                List.of("LlamaForCausalLM"),
                hiddenSize,
                intermediateSize,
                layers,
                attentionHeads,
                keyValueHeads,
                headDim,
                vocabSize,
                maxPositionEmbeddings,
                rmsNormEps,
                ropeTheta,
                hiddenAct,
                false,
                false,
                bosTokenId,
                eosTokenId,
                padTokenId,
                tieWordEmbeddings);
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

    private static double doubleValue(Object value, double defaultValue) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof String text && !text.isBlank()) {
            return Double.parseDouble(text);
        }
        return defaultValue;
    }

    private static boolean booleanValue(Object value, boolean defaultValue) {
        if (value instanceof Boolean flag) {
            return flag;
        }
        if (value instanceof String text && !text.isBlank()) {
            return Boolean.parseBoolean(text);
        }
        return defaultValue;
    }

}
