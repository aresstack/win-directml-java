package com.aresstack.windirectml.inference.gemma;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntConsumer;

/**
 * Greedy generation over the device-free Gemma 3 CPU reference forward pass. This is the parity/native
 * reference generator (not the heap-light WARP product path): each step re-runs prefill over the
 * growing sequence, which is simple and correct for short generations. Stops on EOS / end-of-turn or
 * when {@code maxNewTokens} is reached.
 */
public final class Gemma3ReferenceGenerator {

    private final Gemma3Config config;
    private final Gemma3ReferenceForwardPass forwardPass;

    public Gemma3ReferenceGenerator(Gemma3ReferenceWeights weights) {
        this.config = weights.config();
        this.forwardPass = new Gemma3ReferenceForwardPass(weights);
    }

    /** Result of a greedy generation: the newly generated token ids and why it stopped. */
    public record Result(int[] generatedTokenIds, FinishReason finishReason) {
    }

    public enum FinishReason { EOS, MAX_TOKENS }

    public Result generate(int[] promptIds, int maxNewTokens) {
        return generate(promptIds, maxNewTokens, null);
    }

    /**
     * Greedy-generate up to {@code maxNewTokens}, invoking {@code onToken} for each generated id.
     * Stops early on EOS or end-of-turn.
     */
    public Result generate(int[] promptIds, int maxNewTokens, IntConsumer onToken) {
        Objects.requireNonNull(promptIds, "promptIds");
        if (maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be positive");
        }
        int[] seq = Arrays.copyOf(promptIds, promptIds.length);
        int[] generated = new int[maxNewTokens];
        int count = 0;
        FinishReason reason = FinishReason.MAX_TOKENS;
        for (int step = 0; step < maxNewTokens; step++) {
            int next = forwardPass.nextToken(seq);
            generated[count++] = next;
            if (onToken != null) {
                onToken.accept(next);
            }
            if (isStop(next)) {
                reason = FinishReason.EOS;
                break;
            }
            seq = Arrays.copyOf(seq, seq.length + 1);
            seq[seq.length - 1] = next;
        }
        return new Result(Arrays.copyOf(generated, count), reason);
    }

    private boolean isStop(int tokenId) {
        return tokenId == config.eosTokenId();
    }
}
