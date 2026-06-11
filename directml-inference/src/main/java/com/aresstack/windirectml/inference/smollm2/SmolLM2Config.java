package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyConfig;

import java.util.List;

/**
 * Typed subset of the Hugging Face SmolLM2/Llama CausalLM config.json.
 */
public record SmolLM2Config(
        String modelType,
        List<String> architectures,
        int hiddenSize,
        int intermediateSize,
        int numHiddenLayers,
        int numAttentionHeads,
        Integer numKeyValueHeads,
        Integer headDim,
        int vocabSize,
        int maxPositionEmbeddings,
        double rmsNormEps,
        double ropeTheta,
        String hiddenAct,
        boolean attentionBias,
        boolean mlpBias,
        int bosTokenId,
        int eosTokenId,
        Integer padTokenId,
        boolean tieWordEmbeddings
) {
    public SmolLM2Config {
        modelType = modelType == null ? "" : modelType;
        architectures = architectures == null ? List.of() : List.copyOf(architectures);
        hiddenAct = hiddenAct == null ? "" : hiddenAct;
    }

    public int effectiveKeyValueHeads() {
        return numKeyValueHeads == null || numKeyValueHeads <= 0 ? numAttentionHeads : numKeyValueHeads;
    }

    public int effectiveHeadDim() {
        if (headDim != null && headDim > 0) {
            return headDim;
        }
        if (numAttentionHeads <= 0) {
            return 0;
        }
        return hiddenSize / numAttentionHeads;
    }

    public SmolLM2SpecialTokens specialTokens() {
        return new SmolLM2SpecialTokens(bosTokenId, eosTokenId, padTokenId);
    }

    /**
     * Family-neutral shape view for the shared {@code decoderonly} runtime. The SmolLM2 record cannot implement
     * {@link DecoderOnlyConfig} directly (its key-value-head / head-dim components are nullable {@link Integer} and its
     * epsilons are {@code double}), so this is the explicit ModelConfig adapter point: it resolves the effective GQA
     * head counts / head dim and narrows the epsilons to {@code float}.
     */
    public DecoderOnlyConfig toDecoderOnlyConfig() {
        return DecoderOnlyConfig.of(
                numHiddenLayers,
                hiddenSize,
                intermediateSize,
                numAttentionHeads,
                effectiveKeyValueHeads(),
                effectiveHeadDim(),
                maxPositionEmbeddings,
                vocabSize,
                (float) rmsNormEps,
                (float) ropeTheta);
    }
}
