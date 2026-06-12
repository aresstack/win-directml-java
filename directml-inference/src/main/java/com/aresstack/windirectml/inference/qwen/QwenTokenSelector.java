package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGeneratedTokens;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyGreedyTokenSelector;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyStopTokenPolicy;
import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyTokenSelector;

/**
 * Binds Qwen's greedy-with-repetition-penalty decoding onto the shared {@link DecoderOnlyTokenSelector} seam.
 *
 * <p>The shared {@link DecoderOnlyGreedyTokenSelector} takes the repetition penalty per call (3-arg), while the seam
 * is penalty-free (2-arg). This adapter captures Qwen's configured penalty and delegates, so a future shared
 * generation loop can drive Qwen selection through the same seam SmolLM2 uses. It is preparation only: the existing
 * {@link Qwen2Runtime} decode loop is unchanged and still calls the greedy selector directly.</p>
 */
public final class QwenTokenSelector implements DecoderOnlyTokenSelector {

    private final DecoderOnlyGreedyTokenSelector greedy;
    private final float repetitionPenalty;

    public QwenTokenSelector(DecoderOnlyStopTokenPolicy stopTokenPolicy, float repetitionPenalty) {
        this.greedy = new DecoderOnlyGreedyTokenSelector(stopTokenPolicy);
        this.repetitionPenalty = Math.max(1.0f, repetitionPenalty);
    }

    /** Build a selector from Qwen's stop tokens and the given repetition penalty. */
    public static QwenTokenSelector greedy(float repetitionPenalty) {
        return new QwenTokenSelector(QwenStopTokenPolicy.asDecoderOnlyPolicy(), repetitionPenalty);
    }

    @Override
    public int selectNextToken(float[] logits, DecoderOnlyGeneratedTokens generatedTokens) {
        return greedy.selectNextToken(logits, generatedTokens, repetitionPenalty);
    }

    @Override
    public boolean shouldStop(int tokenId) {
        return greedy.shouldStop(tokenId);
    }
}
