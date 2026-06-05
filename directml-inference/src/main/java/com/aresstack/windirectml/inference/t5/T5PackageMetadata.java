package com.aresstack.windirectml.inference.t5;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * T5-specific metadata written into a wdmlpack manifest.
 */
public final class T5PackageMetadata {
    public static final String MODEL_FAMILY = "t5";
    public static final String ARCHITECTURE = "encoder-decoder";

    private final int dModel;
    private final int dKv;
    private final int dFf;
    private final int numHeads;
    private final int encoderLayers;
    private final int decoderLayers;
    private final int vocabSize;
    private final int relativeAttentionBuckets;
    private final int relativeAttentionMaxDistance;
    private final String feedForwardProjection;
    private final boolean tieWordEmbeddings;
    private final T5SpecialTokens specialTokens;

    private T5PackageMetadata(T5Config config) {
        Objects.requireNonNull(config, "config");
        this.dModel = config.modelSize();
        this.dKv = config.keyValueSize();
        this.dFf = config.feedForwardSize();
        this.numHeads = config.attentionHeads();
        this.encoderLayers = config.encoderLayers();
        this.decoderLayers = config.effectiveDecoderLayers();
        this.vocabSize = config.vocabSize();
        this.relativeAttentionBuckets = config.relativeAttentionBuckets();
        this.relativeAttentionMaxDistance = config.relativeAttentionMaxDistance();
        this.feedForwardProjection = config.effectiveFeedForwardProjection();
        this.tieWordEmbeddings = config.usesTiedWordEmbeddings();
        this.specialTokens = config.specialTokens();
    }

    public static T5PackageMetadata from(T5Config config) {
        return new T5PackageMetadata(config);
    }

    public int dModel() {
        return dModel;
    }

    public int dKv() {
        return dKv;
    }

    public int dFf() {
        return dFf;
    }

    public int numHeads() {
        return numHeads;
    }

    public int encoderLayers() {
        return encoderLayers;
    }

    public int decoderLayers() {
        return decoderLayers;
    }

    public int vocabSize() {
        return vocabSize;
    }

    public int relativeAttentionBuckets() {
        return relativeAttentionBuckets;
    }

    public int relativeAttentionMaxDistance() {
        return relativeAttentionMaxDistance;
    }

    public String feedForwardProjection() {
        return feedForwardProjection;
    }

    public boolean tieWordEmbeddings() {
        return tieWordEmbeddings;
    }

    public T5SpecialTokens specialTokens() {
        return specialTokens;
    }

    public Map<String, Object> toManifest() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("modelFamily", MODEL_FAMILY);
        out.put("architecture", ARCHITECTURE);
        out.put("dModel", dModel);
        out.put("dKv", dKv);
        out.put("dFf", dFf);
        out.put("numHeads", numHeads);
        out.put("encoderLayers", encoderLayers);
        out.put("decoderLayers", decoderLayers);
        out.put("vocabSize", vocabSize);
        out.put("relativeAttentionBuckets", relativeAttentionBuckets);
        out.put("relativeAttentionMaxDistance", relativeAttentionMaxDistance);
        out.put("feedForwardProjection", feedForwardProjection);
        out.put("tieWordEmbeddings", tieWordEmbeddings);
        out.put("padTokenId", specialTokens.padTokenId());
        out.put("eosTokenId", specialTokens.eosTokenId());
        out.put("decoderStartTokenId", specialTokens.decoderStartTokenId());
        return out;
    }
}
