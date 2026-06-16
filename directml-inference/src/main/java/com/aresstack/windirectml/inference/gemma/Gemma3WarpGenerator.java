package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsNativeException;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.IntConsumer;

/**
 * Native WARP greedy generation for Gemma 3 (GEMMA-WARP-10b): prefill → select → decodeNext → repeat,
 * stopping on a stop token or {@code maxNewTokens}.
 *
 * <p>Built on the verified {@link Gemma3WarpDecodeSession} (prefill fills the KV cache, each step reuses
 * it). The stop token ends generation and is excluded from the visible output, and is <b>not</b> streamed
 * — so the {@code onToken} callback receives exactly {@link Gemma3GenerationResult#generatedTokenIds}.
 * The session is owned by the caller; {@link #generate} re-prefills (resetting the cache) each call.</p>
 */
public final class Gemma3WarpGenerator {

    private final Gemma3WarpDecodeSession session;
    private final Gemma3TokenSelector selector;
    private final Gemma3StopTokenPolicy stopPolicy;
    private final Gemma3WarpExecutionMode executionMode;

    public Gemma3WarpGenerator(Gemma3WarpDecodeSession session, Gemma3StopTokenPolicy stopPolicy) {
        this(session, Gemma3TokenSelector.greedy(), stopPolicy);
    }

    public Gemma3WarpGenerator(Gemma3WarpDecodeSession session, Gemma3TokenSelector selector,
                               Gemma3StopTokenPolicy stopPolicy) {
        this(session, selector, stopPolicy, Gemma3WarpExecutionMode.SYNC);
    }

    /** Convenience: greedy selection with an explicit execution mode (GEMMA-WARP-13b-4). */
    public Gemma3WarpGenerator(Gemma3WarpDecodeSession session, Gemma3StopTokenPolicy stopPolicy,
                               Gemma3WarpExecutionMode executionMode) {
        this(session, Gemma3TokenSelector.greedy(), stopPolicy, executionMode);
    }

    public Gemma3WarpGenerator(Gemma3WarpDecodeSession session, Gemma3TokenSelector selector,
                               Gemma3StopTokenPolicy stopPolicy, Gemma3WarpExecutionMode executionMode) {
        this.session = Objects.requireNonNull(session, "session");
        this.selector = Objects.requireNonNull(selector, "selector");
        this.stopPolicy = Objects.requireNonNull(stopPolicy, "stopPolicy");
        this.executionMode = Objects.requireNonNull(executionMode, "executionMode");
    }

    public Gemma3WarpExecutionMode executionMode() {
        return executionMode;
    }

    public Gemma3GenerationResult generate(Gemma3GenerationRequest request) throws WindowsNativeException {
        return generate(request, null);
    }

    /**
     * Generate up to {@code maxNewTokens}, invoking {@code onToken} for each visible token (the stop
     * token is not streamed). Re-prefills the session's KV cache from {@code request.promptIds()}.
     */
    public Gemma3GenerationResult generate(Gemma3GenerationRequest request, IntConsumer onToken)
            throws WindowsNativeException {
        Objects.requireNonNull(request, "request");
        int[] prompt = request.promptIds();
        int maxNew = request.maxNewTokens();

        boolean resident = executionMode.isResident();
        // GEMMA-WARP-13e: resident prefill runs the batched path (whole prompt at once); sync path unchanged.
        float[] logits = resident ? session.prefillResidentBatched(prompt) : session.prefill(prompt);
        List<Integer> generated = new ArrayList<>();
        Gemma3GenerationResult.FinishReason reason = Gemma3GenerationResult.FinishReason.MAX_TOKENS;

        while (generated.size() < maxNew) {
            int next = selector.select(logits);
            if (stopPolicy.isStop(next)) {
                reason = Gemma3GenerationResult.FinishReason.STOP_TOKEN;
                break;
            }
            generated.add(next);
            if (onToken != null) {
                onToken.accept(next);
            }
            if (generated.size() >= maxNew) {
                break; // reason stays MAX_TOKENS; avoid an extra decode step
            }
            logits = resident ? session.decodeNextResident(next) : session.decodeNext(next);
        }

        int[] gen = generated.stream().mapToInt(Integer::intValue).toArray();
        int[] full = new int[prompt.length + gen.length];
        System.arraycopy(prompt, 0, full, 0, prompt.length);
        System.arraycopy(gen, 0, full, prompt.length, gen.length);
        return new Gemma3GenerationResult(gen, full, reason, prompt.length, gen.length);
    }
}
