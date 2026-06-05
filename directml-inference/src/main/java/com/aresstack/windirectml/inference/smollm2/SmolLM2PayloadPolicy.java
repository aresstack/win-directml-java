package com.aresstack.windirectml.inference.smollm2;

/**
 * SmolLM2 package payload policy.
 */
public enum SmolLM2PayloadPolicy {
    MANIFEST_ONLY(false),
    DENSE_PAYLOAD(true);

    private final boolean payloadIncluded;

    SmolLM2PayloadPolicy(boolean payloadIncluded) {
        this.payloadIncluded = payloadIncluded;
    }

    public boolean payloadIncluded() {
        return payloadIncluded;
    }
}
