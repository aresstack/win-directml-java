package com.aresstack.windirectml.inference.smollm2;

import java.util.Locale;

/**
 * Describes the SmolLM2 family without creating compiler, tokenizer, or runtime services.
 */
public final class SmolLM2ModelFamily {

    public String id() {
        return "smollm2";
    }

    public String displayName() {
        return "SmolLM2";
    }

    public boolean supports(SmolLM2Config config) {
        if (config == null) {
            return false;
        }
        return "llama".equalsIgnoreCase(config.modelType())
                && config.architectures().stream().anyMatch("LlamaForCausalLM"::equals)
                && "silu".equals(config.hiddenAct().toLowerCase(Locale.ROOT))
                && !config.attentionBias();
    }

    public SmolLM2Architecture architecture(SmolLM2Config config) {
        if (!supports(config)) {
            throw new IllegalArgumentException("unsupported SmolLM2/Llama CausalLM config");
        }
        return SmolLM2Architecture.from(config);
    }
}
