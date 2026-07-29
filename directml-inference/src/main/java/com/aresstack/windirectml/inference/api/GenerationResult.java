package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;

/**
 * The outcome of a completed generation call.
 *
 * @param text            the full generated text (concatenation of all streamed token pieces)
 * @param finishReason    why generation stopped (stop/eos token vs. max new tokens)
 * @param promptTokenCount number of prompt tokens fed to the model
 * @param generatedTokenCount number of tokens the model produced
 * @param backend         the backend the model actually ran on
 * @param runtimeModelId  the catalog runtime model id that produced this result
 */
public record GenerationResult(
        String text,
        GenerationFinishReason finishReason,
        int promptTokenCount,
        int generatedTokenCount,
        CatalogBackend backend,
        String runtimeModelId) {

    public GenerationResult {
        if (text == null) {
            text = "";
        }
    }
}
