package com.aresstack.windirectml.config.generation;

import java.util.Objects;

/**
 * Shared, model-agnostic request for text generation.
 *
 * <p>All generation backends (Phi-3, Qwen, future Seq2Seq) accept this
 * request type. Model-specific prompt formatting is handled by the
 * implementation, not the caller.
 *
 * <p>Java-8 compatible (builder pattern, no records).
 */
public final class GenerationRequest {

    private final String userPrompt;
    private final String systemPrompt;
    private final int maxTokens;
    private final SamplerConfig sampler;
    private final StopTokenPolicy stopPolicy;

    private GenerationRequest(Builder builder) {
        this.userPrompt = builder.userPrompt;
        this.systemPrompt = builder.systemPrompt;
        this.maxTokens = builder.maxTokens;
        this.sampler = builder.sampler;
        this.stopPolicy = builder.stopPolicy;
    }

    /** The user-facing prompt text. Never null. */
    public String userPrompt() { return userPrompt; }

    /** Optional system prompt. May be empty but never null. */
    public String systemPrompt() { return systemPrompt; }

    /** Maximum number of tokens to generate; &le; 0 means backend default. */
    public int maxTokens() { return maxTokens; }

    /** Sampling configuration. Never null (defaults to greedy). */
    public SamplerConfig sampler() { return sampler; }

    /** Stop-token policy. Never null (defaults to EOS-only). */
    public StopTokenPolicy stopPolicy() { return stopPolicy; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String userPrompt = "";
        private String systemPrompt = "";
        private int maxTokens = 0;
        private SamplerConfig sampler = SamplerConfig.greedy();
        private StopTokenPolicy stopPolicy = StopTokenPolicy.eosOnly();

        public Builder userPrompt(String userPrompt) {
            this.userPrompt = Objects.requireNonNull(userPrompt, "userPrompt");
            return this;
        }

        public Builder systemPrompt(String systemPrompt) {
            this.systemPrompt = Objects.requireNonNull(systemPrompt, "systemPrompt");
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public Builder sampler(SamplerConfig sampler) {
            this.sampler = Objects.requireNonNull(sampler, "sampler");
            return this;
        }

        public Builder stopPolicy(StopTokenPolicy stopPolicy) {
            this.stopPolicy = Objects.requireNonNull(stopPolicy, "stopPolicy");
            return this;
        }

        public GenerationRequest build() {
            return new GenerationRequest(this);
        }
    }

    @Override
    public String toString() {
        return "GenerationRequest{userPrompt='" +
                (userPrompt.length() > 40 ? userPrompt.substring(0, 40) + "\u2026" : userPrompt) +
                "', maxTokens=" + maxTokens + ", sampler=" + sampler + "}";
    }
}
