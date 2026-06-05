package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Objects;
import java.util.Random;

/**
 * Selects the next token using temperature, top-k and nucleus sampling.
 */
final class SmolLM2SamplingTokenSampler implements SmolLM2TokenSampler {

    private static final double MIN_TEMPERATURE = 1.0e-8d;

    private final DecoderOnlyStopTokenPolicy stopTokenPolicy;
    private final SmolLM2GenerationOptions options;
    private final Random random;

    SmolLM2SamplingTokenSampler(DecoderOnlyStopTokenPolicy stopTokenPolicy,
                                SmolLM2GenerationOptions options,
                                Random random) {
        this.stopTokenPolicy = Objects.requireNonNull(stopTokenPolicy, "stopTokenPolicy");
        this.options = Objects.requireNonNull(options, "options");
        this.random = Objects.requireNonNull(random, "random");
    }

    @Override
    public int selectNextToken(float[] logits, DecoderOnlyGeneratedTokens generatedTokens) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(generatedTokens, "generatedTokens");
        float[] adjustedLogits = Arrays.copyOf(logits, logits.length);
        DecoderOnlyMath.applyRepetitionPenalty(
                adjustedLogits,
                generatedTokens.backingArray(),
                generatedTokens.count(),
                (float) options.effectiveRepetitionPenalty());
        double[] probabilities = softmax(adjustedLogits, Math.max(options.effectiveTemperature(), MIN_TEMPERATURE));
        applyTopK(probabilities, options.effectiveTopK());
        applyTopP(probabilities, options.effectiveTopP());
        normalize(probabilities);
        return sample(probabilities);
    }

    @Override
    public boolean shouldStop(int tokenId) {
        return stopTokenPolicy.shouldStop(tokenId);
    }

    private static double[] softmax(float[] logits, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (float logit : logits) {
            double scaled = logit / temperature;
            if (scaled > max) {
                max = scaled;
            }
        }
        double[] probabilities = new double[logits.length];
        double sum = 0.0d;
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = Math.exp((logits[i] / temperature) - max);
            sum += probabilities[i];
        }
        if (sum == 0.0d || Double.isNaN(sum) || Double.isInfinite(sum)) {
            throw new IllegalStateException("Cannot sample from invalid logits");
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
        return probabilities;
    }

    private static void applyTopK(double[] probabilities, int topK) {
        if (topK <= 0 || topK >= probabilities.length) {
            return;
        }
        Integer[] indexes = sortedIndexesByProbability(probabilities);
        for (int i = topK; i < indexes.length; i++) {
            probabilities[indexes[i]] = 0.0d;
        }
    }

    private static void applyTopP(double[] probabilities, double topP) {
        if (topP >= 1.0d) {
            return;
        }
        Integer[] indexes = sortedIndexesByProbability(probabilities);
        double cumulative = 0.0d;
        boolean keepCurrent = true;
        for (int index : indexes) {
            if (!keepCurrent) {
                probabilities[index] = 0.0d;
                continue;
            }
            cumulative += probabilities[index];
            keepCurrent = cumulative < topP;
        }
    }

    private static Integer[] sortedIndexesByProbability(double[] probabilities) {
        Integer[] indexes = new Integer[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, Comparator.comparingDouble((Integer index) -> probabilities[index]).reversed());
        return indexes;
    }

    private static void normalize(double[] probabilities) {
        double sum = 0.0d;
        for (double probability : probabilities) {
            sum += probability;
        }
        if (sum <= 0.0d || Double.isNaN(sum) || Double.isInfinite(sum)) {
            throw new IllegalStateException("Cannot normalize an empty sampling distribution");
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
    }

    private int sample(double[] probabilities) {
        double threshold = random.nextDouble();
        double cumulative = 0.0d;
        for (int i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (threshold <= cumulative) {
                return i;
            }
        }
        return probabilities.length - 1;
    }
}
