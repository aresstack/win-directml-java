package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsNativeException;

/**
 * Explicit shader/kernel warm-up for the Gemma native WARP resident path (GEMMA-WARP-13f): runs one short
 * batched prefill plus a few resident decode steps so every lazily-compiled shader (the batched-matmul
 * prefill shader, resident matvec, RMSNorm, QK-norm, RoPE, attention scores/softmax/value, GeGLU, the tied
 * LM head, KV append, row-copy) is compiled and run once <b>before</b> a measured region.
 *
 * <p>Pure timing hygiene — it changes no model logic and leaves the session ready for a fresh prefill (the
 * next {@code prefillResident*} resets the KV cache). Use it to keep the one-time lazy-compile cost out of
 * warm prefill/decode measurements.</p>
 */
public final class Gemma3WarpWarmup {

    private Gemma3WarpWarmup() {
    }

    /** A tiny valid prompt (bos + two ids) for warm-up; content is irrelevant, only shape matters. */
    public static int[] defaultWarmupPrompt(Gemma3Config config) {
        int bos = config.bosTokenId();
        int a = Math.min(100, config.vocabSize() - 1);
        int b = Math.min(200, config.vocabSize() - 1);
        return new int[]{bos, a, b};
    }

    /**
     * Warm up {@code session} by running one batched prefill of {@code warmupPromptIds} and {@code decodeSteps}
     * resident decode steps. Returns the number of decode steps actually run (for logging). The session's KV
     * cache is left populated by this throwaway run; a subsequent {@code prefillResident*} resets it.
     */
    public static int warmUp(Gemma3WarpDecodeSession session, int[] warmupPromptIds, int decodeSteps)
            throws WindowsNativeException {
        int next = session.prefillNextTokenResidentBatched(warmupPromptIds);
        int steps = 0;
        for (int i = 0; i < decodeSteps; i++) {
            next = session.decodeNextTokenResident(next);
            steps++;
        }
        return steps;
    }
}
