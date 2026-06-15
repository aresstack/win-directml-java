package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;

/**
 * Selects the next token id from a vocab-sized logits vector (GEMMA-WARP-10b). This slice only ships the
 * greedy selector ({@code argmax}); sampling strategies can be added later behind the same seam.
 */
@FunctionalInterface
public interface Gemma3TokenSelector {

    int select(float[] logits);

    /** Greedy selection: the highest-logit token id. */
    static Gemma3TokenSelector greedy() {
        return DecoderOnlyMath::argmax;
    }
}
