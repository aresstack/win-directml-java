package com.aresstack.windirectml.sidecar.client.validation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class ModelExpectation {
    private final String label;
    private final List<String> requiredFiles;
    private final List<List<String>> eitherOfFiles;
    private final Integer hiddenSize;
    private final Integer layers;
    private final Integer heads;
    private final String tokenizerType;
    private final boolean ready;
    private final String notReadyReason;

    private ModelExpectation(Builder builder) {
        this.label = builder.label;
        this.requiredFiles = Collections.unmodifiableList(new ArrayList<String>(builder.requiredFiles));
        List<List<String>> groups = new ArrayList<List<String>>();
        for (List<String> group : builder.eitherOfFiles) {
            groups.add(Collections.unmodifiableList(new ArrayList<String>(group)));
        }
        this.eitherOfFiles = Collections.unmodifiableList(groups);
        this.hiddenSize = builder.hiddenSize;
        this.layers = builder.layers;
        this.heads = builder.heads;
        this.tokenizerType = builder.tokenizerType;
        this.ready = builder.ready;
        this.notReadyReason = builder.notReadyReason;
    }

    public String getLabel() { return label; }
    public List<String> getRequiredFiles() { return requiredFiles; }
    public List<List<String>> getEitherOfFiles() { return eitherOfFiles; }
    public Integer getHiddenSize() { return hiddenSize; }
    public Integer getLayers() { return layers; }
    public Integer getHeads() { return heads; }
    public String getTokenizerType() { return tokenizerType; }
    public boolean isReady() { return ready; }
    public String getNotReadyReason() { return notReadyReason; }

    public static Builder builder(String label) { return new Builder(label); }

    public static final class Builder {
        private final String label;
        private final Set<String> requiredFiles = new LinkedHashSet<String>();
        private final List<List<String>> eitherOfFiles = new ArrayList<List<String>>();
        private Integer hiddenSize;
        private Integer layers;
        private Integer heads;
        private String tokenizerType;
        private boolean ready = true;
        private String notReadyReason;

        private Builder(String label) { this.label = label == null ? "model" : label; }

        public Builder require(String... files) {
            requiredFiles.addAll(Arrays.asList(files));
            return this;
        }

        public Builder eitherOf(String... files) {
            eitherOfFiles.add(Arrays.asList(files));
            return this;
        }

        public Builder shape(int hiddenSize, int layers, int heads) {
            this.hiddenSize = hiddenSize;
            this.layers = layers;
            this.heads = heads;
            return this;
        }

        public Builder tokenizerType(String tokenizerType) {
            this.tokenizerType = tokenizerType;
            return this;
        }

        public Builder notReady(String reason) {
            this.ready = false;
            this.notReadyReason = reason;
            return this;
        }

        public ModelExpectation build() { return new ModelExpectation(this); }
    }
}
