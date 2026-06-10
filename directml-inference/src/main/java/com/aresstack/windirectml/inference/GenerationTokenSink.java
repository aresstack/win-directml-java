package com.aresstack.windirectml.inference;

/**
 * Receives generated tokens while a model is running.
 */
public interface GenerationTokenSink {

    GenerationTokenSink IGNORE = new GenerationTokenSink() {
        @Override
        public void onToken(GeneratedToken token) {
            // Ignore token events.
        }

        @Override
        public void onCompleted(InferenceResult result) {
            // Ignore completion events.
        }
    };

    void onToken(GeneratedToken token);

    default void onCompleted(InferenceResult result) {
        // Allow sinks to ignore completion events.
    }
}
