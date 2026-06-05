package com.aresstack.windirectml.inference.smollm2;

/**
 * Initial SmolLM2 package policy: write manifest diagnostics only.
 */
public enum SmolLM2PayloadPolicy {
    MANIFEST_ONLY(false);

    private final boolean payloadIncluded;

    SmolLM2PayloadPolicy(boolean payloadIncluded) {
        this.payloadIncluded = payloadIncluded;
    }

    public boolean payloadIncluded() {
        return payloadIncluded;
    }
}
