package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;

import java.util.Objects;
import java.util.Random;

/**
 * Creates token samplers from generation options.
 */
final class SmolLM2TokenSamplerFactory {

    private final DecoderOnlyStopTokenPolicy stopTokenPolicy;

    SmolLM2TokenSamplerFactory(DecoderOnlyStopTokenPolicy stopTokenPolicy) {
        this.stopTokenPolicy = Objects.requireNonNull(stopTokenPolicy, "stopTokenPolicy");
    }

    SmolLM2TokenSampler create(SmolLM2GenerationOptions options) {
        SmolLM2GenerationOptions effectiveOptions = options == null ? SmolLM2GenerationOptions.greedy() : options;
        if (!effectiveOptions.usesSampling()) {
            return new SmolLM2GreedyTokenSampler(stopTokenPolicy, effectiveOptions.effectiveRepetitionPenalty());
        }
        Long seed = effectiveOptions.randomSeed();
        Random random = seed == null ? new Random() : new Random(seed);
        return new SmolLM2SamplingTokenSampler(stopTokenPolicy, effectiveOptions, random);
    }
}
