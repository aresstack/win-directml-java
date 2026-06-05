package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Immutable architecture snapshot for the curated T5/CodeT5 family.
 */
public final class T5Architecture {
    private final int modelSize;
    private final int keyValueSize;
    private final int feedForwardSize;
    private final int attentionHeads;
    private final int attentionInnerSize;
    private final int encoderLayers;
    private final int decoderLayers;
    private final int vocabSize;
    private final int relativeAttentionBuckets;
    private final int relativeAttentionMaxDistance;
    private final boolean gatedFeedForward;
    private final boolean tiedWordEmbeddings;

    private T5Architecture(T5Config config) {
        Objects.requireNonNull(config, "config");
        this.modelSize = requirePositive("dModel", config.modelSize());
        this.keyValueSize = requirePositive("dKv", config.keyValueSize());
        this.feedForwardSize = requirePositive("dFf", config.feedForwardSize());
        this.attentionHeads = requirePositive("numHeads", config.attentionHeads());
        this.attentionInnerSize = requirePositive("attentionInnerSize", config.attentionInnerSize());
        this.encoderLayers = requirePositive("encoderLayers", config.encoderLayers());
        this.decoderLayers = requirePositive("decoderLayers", config.effectiveDecoderLayers());
        this.vocabSize = requirePositive("vocabSize", config.vocabSize());
        this.relativeAttentionBuckets = requireNonNegative("relativeAttentionBuckets", config.relativeAttentionBuckets());
        this.relativeAttentionMaxDistance = requireNonNegative("relativeAttentionMaxDistance", config.relativeAttentionMaxDistance());
        this.gatedFeedForward = config.usesGatedFeedForward();
        this.tiedWordEmbeddings = config.usesTiedWordEmbeddings();
    }

    public static T5Architecture from(T5Config config) {
        return new T5Architecture(config);
    }

    public int modelSize() {
        return modelSize;
    }

    public int keyValueSize() {
        return keyValueSize;
    }

    public int feedForwardSize() {
        return feedForwardSize;
    }

    public int attentionHeads() {
        return attentionHeads;
    }

    public int attentionInnerSize() {
        return attentionInnerSize;
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

    public boolean usesGatedFeedForward() {
        return gatedFeedForward;
    }

    public boolean usesTiedWordEmbeddings() {
        return tiedWordEmbeddings;
    }

    private static int requirePositive(String name, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException("T5 architecture requires positive " + name + ": " + value);
        }
        return value;
    }

    private static int requireNonNegative(String name, int value) {
        if (value < 0) {
            throw new IllegalArgumentException("T5 architecture requires non-negative " + name + ": " + value);
        }
        return value;
    }
}
