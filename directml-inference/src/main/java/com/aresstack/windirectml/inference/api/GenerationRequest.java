package com.aresstack.windirectml.inference.api;

import java.util.Objects;

/**
 * A single generation call. Immutable; build with {@link #builder(String)}.
 *
 * <p>The runtime performs greedy/deterministic decoding: given the same package, backend, and
 * request it produces the same output. The prompt is neutral — the family adapter applies the
 * model's chat template / task prefix (e.g. ChatML for Qwen/SmolLM2, {@code <start_of_turn>} for
 * Gemma, {@code "summarize: "} for T5). Seq2seq callers put the full task string (e.g.
 * {@code "summarize: ..."}) in {@code userPrompt} and leave {@code systemPrompt} null.
 */
public final class GenerationRequest {

    /** Default cap on newly generated tokens when the caller does not set one. */
    public static final int DEFAULT_MAX_NEW_TOKENS = 256;

    private final String userPrompt;
    private final String systemPrompt;
    private final int maxNewTokens;

    private GenerationRequest(Builder b) {
        this.userPrompt = b.userPrompt;
        this.systemPrompt = b.systemPrompt;
        this.maxNewTokens = b.maxNewTokens;
    }

    /** The user/task prompt (required, non-blank). */
    public String userPrompt() {
        return userPrompt;
    }

    /** Optional system prompt; {@code null} when not set. Ignored by seq2seq families. */
    public String systemPrompt() {
        return systemPrompt;
    }

    /** Maximum number of new tokens to generate. */
    public int maxNewTokens() {
        return maxNewTokens;
    }

    public static Builder builder(String userPrompt) {
        return new Builder(userPrompt);
    }

    public static GenerationRequest of(String userPrompt) {
        return builder(userPrompt).build();
    }

    public static final class Builder {
        private final String userPrompt;
        private String systemPrompt;
        private int maxNewTokens = DEFAULT_MAX_NEW_TOKENS;

        private Builder(String userPrompt) {
            this.userPrompt = userPrompt;
        }

        public Builder systemPrompt(String systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder maxNewTokens(int maxNewTokens) {
            this.maxNewTokens = maxNewTokens;
            return this;
        }

        public GenerationRequest build() {
            if (userPrompt == null || userPrompt.trim().isEmpty()) {
                throw new GenerationException(GenerationErrorCode.INVALID_REQUEST,
                        "userPrompt must be non-blank");
            }
            if (maxNewTokens <= 0) {
                throw new GenerationException(GenerationErrorCode.INVALID_REQUEST,
                        "maxNewTokens must be > 0, was " + maxNewTokens);
            }
            return new GenerationRequest(this);
        }
    }

    @Override
    public String toString() {
        return "GenerationRequest{maxNewTokens=" + maxNewTokens
                + ", hasSystemPrompt=" + (systemPrompt != null)
                + ", userPromptLength=" + (userPrompt == null ? 0 : userPrompt.length()) + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof GenerationRequest)) {
            return false;
        }
        GenerationRequest that = (GenerationRequest) o;
        return maxNewTokens == that.maxNewTokens
                && Objects.equals(userPrompt, that.userPrompt)
                && Objects.equals(systemPrompt, that.systemPrompt);
    }

    @Override
    public int hashCode() {
        return Objects.hash(userPrompt, systemPrompt, maxNewTokens);
    }
}
