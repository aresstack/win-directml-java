package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SmolLM2TokenSamplerTest {

    @Test
    void greedyOptionsKeepTheExistingDefault() {
        SmolLM2GenerationOptions options = SmolLM2GenerationOptions.greedy();

        assertEquals(0.0d, options.effectiveTemperature());
        assertEquals(0, options.effectiveTopK());
        assertEquals(1.0d, options.effectiveTopP());
        assertEquals(1.0d, options.effectiveRepetitionPenalty());
    }

    @Test
    void generationOptionsRejectInvalidSamplingValues() {
        assertThrows(IllegalArgumentException.class,
                () -> new SmolLM2GenerationOptions(-0.1d, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new SmolLM2GenerationOptions(1.0d, -1, null));
        assertThrows(IllegalArgumentException.class,
                () -> new SmolLM2GenerationOptions(1.0d, null, 0.0d));
        assertThrows(IllegalArgumentException.class,
                () -> new SmolLM2GenerationOptions(1.0d, null, 1.1d));
        assertThrows(IllegalArgumentException.class,
                () -> new SmolLM2GenerationOptions(1.0d, null, null, null, 0.0d));
    }

    @Test
    void samplingWithTopKOneSelectsOnlyHighestLogitToken() {
        SmolLM2TokenSampler sampler = sampler(SmolLM2GenerationOptions.sampling(1.0d, 1, null, 7L));

        int tokenId = sampler.selectNextToken(new float[]{0.0f, 1.0f, 5.0f, 2.0f}, new DecoderOnlyGeneratedTokens(4));

        assertEquals(2, tokenId);
    }

    @Test
    void samplingWithTopPBelowBestTokenProbabilityKeepsTheBestToken() {
        SmolLM2TokenSampler sampler = sampler(SmolLM2GenerationOptions.sampling(1.0d, null, 0.25d, 7L));

        int tokenId = sampler.selectNextToken(new float[]{10.0f, 1.0f, 0.0f}, new DecoderOnlyGeneratedTokens(4));

        assertEquals(0, tokenId);
    }

    @Test
    void factoryKeepsGreedySelectionWhenTemperatureIsZero() {
        SmolLM2TokenSampler sampler = sampler(new SmolLM2GenerationOptions(0.0d, 1, 0.5d));

        int tokenId = sampler.selectNextToken(new float[]{0.0f, 1.0f, 3.0f}, new DecoderOnlyGeneratedTokens(4));

        assertEquals(2, tokenId);
    }

    private static SmolLM2TokenSampler sampler(SmolLM2GenerationOptions options) {
        SmolLM2TokenSamplerFactory factory = new SmolLM2TokenSamplerFactory(
                DecoderOnlyStopTokenPolicy.fromTokenIds(List.of(2)));
        return factory.create(options);
    }
}
