package com.aresstack.windirectml.inference.api;

/**
 * A single streamed generation step.
 *
 * @param text    the decoded text piece for this step (may be empty for a non-printing token)
 * @param tokenId the model vocabulary id that was emitted
 * @param index   zero-based position of this token within the generated sequence
 */
public record GenerationToken(String text, int tokenId, int index) {

    public GenerationToken {
        if (text == null) {
            text = "";
        }
    }
}
