package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.SourceTensorDataType;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Creates a deterministic resource and kernel skeleton for the future SmolLM2 WARP runtime.
 */
public final class SmolLM2WarpRuntimePlanner {

    public static final int DEFAULT_ALIGNMENT_BYTES = 256;
    public static final int DEFAULT_ACTIVATION_BYTES = Float.BYTES;

    private final int alignmentBytes;
    private final int activationBytes;

    public SmolLM2WarpRuntimePlanner() {
        this(DEFAULT_ALIGNMENT_BYTES, DEFAULT_ACTIVATION_BYTES);
    }

    public SmolLM2WarpRuntimePlanner(int alignmentBytes, int activationBytes) {
        if (alignmentBytes <= 0) {
            throw new IllegalArgumentException("alignmentBytes must be positive");
        }
        if (activationBytes <= 0) {
            throw new IllegalArgumentException("activationBytes must be positive");
        }
        this.alignmentBytes = alignmentBytes;
        this.activationBytes = activationBytes;
    }

    public SmolLM2WarpRuntimePlan plan(SmolLM2Weights weights, int sequenceLength) {
        Objects.requireNonNull(weights, "weights");
        if (sequenceLength <= 0) {
            throw new IllegalArgumentException("sequenceLength must be positive");
        }

        List<String> warnings = new ArrayList<>();
        collectCompatibilityWarnings(weights.config(), warnings);

        List<SmolLM2WarpBufferEntry> entries = new ArrayList<>();
        long weightBytes = appendWeightEntries(entries, weights, 0L);
        long kvBytes = appendKvCacheEntries(entries, weights.config(), sequenceLength, weightBytes);
        long scratchBytes = appendScratchEntries(entries, weights.config(), sequenceLength, weightBytes + kvBytes);

        SmolLM2WarpBufferPlan bufferPlan = new SmolLM2WarpBufferPlan(
                entries,
                weightBytes,
                kvBytes,
                scratchBytes,
                alignmentBytes);
        return new SmolLM2WarpRuntimePlan(
                weights.config(),
                sequenceLength,
                bufferPlan,
                createKernelPlan(weights.config()),
                warnings);
    }

    private long appendWeightEntries(List<SmolLM2WarpBufferEntry> entries,
                                     SmolLM2Weights weights,
                                     long baseOffset) {
        List<SmolLM2WeightTensor> tensors = new ArrayList<>();
        tensors.add(weights.tokenEmbedding());
        tensors.add(weights.finalNorm());
        if (!weights.lmHeadTiedToEmbedding()) {
            tensors.add(weights.lmHead());
        }
        for (SmolLM2LayerWeights layer : weights.layers()) {
            tensors.addAll(layer.tensors().values());
        }
        tensors.sort(Comparator.comparing(SmolLM2WeightTensor::tensorName));

        long offset = baseOffset;
        for (SmolLM2WeightTensor tensor : tensors) {
            long alignedLength = align(tensor.rawByteLength());
            entries.add(new SmolLM2WarpBufferEntry(
                    tensor.tensorName(),
                    SmolLM2WarpBufferKind.WEIGHT,
                    offset,
                    tensor.rawByteLength(),
                    alignedLength,
                    tensor.dims(),
                    tensor.dataType(),
                    tensor.role(),
                    tensor.layerIndex()));
            offset += alignedLength;
        }
        return offset - baseOffset;
    }

    private long appendKvCacheEntries(List<SmolLM2WarpBufferEntry> entries,
                                      SmolLM2Config config,
                                      int sequenceLength,
                                      long baseOffset) {
        long offset = baseOffset;
        int kvWidth = Math.multiplyExact(config.effectiveKeyValueHeads(), config.effectiveHeadDim());
        long logicalBytes = Math.multiplyExact(Math.multiplyExact((long) sequenceLength, kvWidth), activationBytes);
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            offset = addGeneratedEntry(entries, "layer.%03d.k_cache".formatted(layer),
                    SmolLM2WarpBufferKind.KV_CACHE, offset, logicalBytes,
                    new long[]{sequenceLength, config.effectiveKeyValueHeads(), config.effectiveHeadDim()}, layer);
            offset = addGeneratedEntry(entries, "layer.%03d.v_cache".formatted(layer),
                    SmolLM2WarpBufferKind.KV_CACHE, offset, logicalBytes,
                    new long[]{sequenceLength, config.effectiveKeyValueHeads(), config.effectiveHeadDim()}, layer);
        }
        return offset - baseOffset;
    }

    private long appendScratchEntries(List<SmolLM2WarpBufferEntry> entries,
                                      SmolLM2Config config,
                                      int sequenceLength,
                                      long baseOffset) {
        long offset = baseOffset;
        offset = addGeneratedEntry(entries, "scratch.hidden_state", SmolLM2WarpBufferKind.SCRATCH,
                offset, bytes(config.hiddenSize()), new long[]{config.hiddenSize()}, -1);
        offset = addGeneratedEntry(entries, "scratch.q", SmolLM2WarpBufferKind.SCRATCH,
                offset, bytes(config.numAttentionHeads() * config.effectiveHeadDim()),
                new long[]{config.numAttentionHeads(), config.effectiveHeadDim()}, -1);
        offset = addGeneratedEntry(entries, "scratch.k", SmolLM2WarpBufferKind.SCRATCH,
                offset, bytes(config.effectiveKeyValueHeads() * config.effectiveHeadDim()),
                new long[]{config.effectiveKeyValueHeads(), config.effectiveHeadDim()}, -1);
        offset = addGeneratedEntry(entries, "scratch.v", SmolLM2WarpBufferKind.SCRATCH,
                offset, bytes(config.effectiveKeyValueHeads() * config.effectiveHeadDim()),
                new long[]{config.effectiveKeyValueHeads(), config.effectiveHeadDim()}, -1);
        offset = addGeneratedEntry(entries, "scratch.attention_scores", SmolLM2WarpBufferKind.SCRATCH,
                offset, bytes(config.numAttentionHeads() * sequenceLength),
                new long[]{config.numAttentionHeads(), sequenceLength}, -1);
        offset = addGeneratedEntry(entries, "scratch.mlp_gate", SmolLM2WarpBufferKind.SCRATCH,
                offset, bytes(config.intermediateSize()), new long[]{config.intermediateSize()}, -1);
        offset = addGeneratedEntry(entries, "scratch.mlp_up", SmolLM2WarpBufferKind.SCRATCH,
                offset, bytes(config.intermediateSize()), new long[]{config.intermediateSize()}, -1);
        return addGeneratedEntry(entries, "scratch.logits", SmolLM2WarpBufferKind.SCRATCH,
                offset, bytes(config.vocabSize()), new long[]{config.vocabSize()}, -1) - baseOffset;
    }

    private long addGeneratedEntry(List<SmolLM2WarpBufferEntry> entries,
                                   String name,
                                   SmolLM2WarpBufferKind kind,
                                   long offset,
                                   long byteLength,
                                   long[] dims,
                                   int layerIndex) {
        long alignedLength = align(byteLength);
        entries.add(new SmolLM2WarpBufferEntry(
                name,
                kind,
                offset,
                byteLength,
                alignedLength,
                dims,
                SourceTensorDataType.FLOAT.onnxCode(),
                null,
                layerIndex));
        return offset + alignedLength;
    }

    private SmolLM2WarpKernelPlan createKernelPlan(SmolLM2Config config) {
        List<SmolLM2WarpKernelStep> steps = new ArrayList<>();
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            steps.add(new SmolLM2WarpKernelStep("rms_norm.input", layer, "hidden_state", "hidden_state_norm"));
            steps.add(new SmolLM2WarpKernelStep("matmul.qkv", layer, "hidden_state_norm", "qkv"));
            steps.add(new SmolLM2WarpKernelStep("rope.apply", layer, "qkv", "qkv_rope"));
            steps.add(new SmolLM2WarpKernelStep("attention.causal", layer, "qkv_rope", "attention_output"));
            steps.add(new SmolLM2WarpKernelStep("matmul.attention_output", layer, "attention_output", "hidden_state"));
            steps.add(new SmolLM2WarpKernelStep("rms_norm.post_attention", layer, "hidden_state", "hidden_state_norm"));
            steps.add(new SmolLM2WarpKernelStep("mlp.gate_up_silu", layer, "hidden_state_norm", "mlp_intermediate"));
            steps.add(new SmolLM2WarpKernelStep("matmul.mlp_down", layer, "mlp_intermediate", "hidden_state"));
        }
        steps.add(new SmolLM2WarpKernelStep("rms_norm.final", -1, "hidden_state", "hidden_state_norm"));
        steps.add(new SmolLM2WarpKernelStep("lm_head", -1, "hidden_state_norm", "logits"));
        return new SmolLM2WarpKernelPlan(steps);
    }

    private void collectCompatibilityWarnings(SmolLM2Config config, List<String> warnings) {
        if (!"silu".equalsIgnoreCase(config.hiddenAct())) {
            warnings.add("SmolLM2 WARP planner currently expects hidden_act=silu but found " + config.hiddenAct());
        }
        if (config.attentionBias()) {
            warnings.add("SmolLM2 WARP planner does not allocate attention bias buffers yet");
        }
        if (config.mlpBias()) {
            warnings.add("SmolLM2 WARP planner does not allocate MLP bias buffers yet");
        }
        if (config.effectiveHeadDim() * config.numAttentionHeads() != config.hiddenSize()) {
            warnings.add("attention head layout does not multiply back to hidden_size");
        }
    }

    private long bytes(int elements) {
        return Math.multiplyExact((long) elements, activationBytes);
    }

    private long align(long byteLength) {
        long remainder = byteLength % alignmentBytes;
        return remainder == 0L ? byteLength : byteLength + alignmentBytes - remainder;
    }
}
