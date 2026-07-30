package com.aresstack.windirectml.inference.api;

/**
 * Receives streamed tokens during {@link GenerationModelHandle#generate(GenerationRequest,
 * GenerationTokenListener)}. Invoked synchronously on the generation thread in emission order.
 */
@FunctionalInterface
public interface GenerationTokenListener {

    /** Called once per generated token, in order. */
    void onToken(GenerationToken token);

    /**
     * Called once after the final token, with the assembled result. Default is a no-op so a listener
     * can be a lambda over {@link #onToken}.
     */
    default void onComplete(GenerationResult result) {
        // no-op by default
    }
}
