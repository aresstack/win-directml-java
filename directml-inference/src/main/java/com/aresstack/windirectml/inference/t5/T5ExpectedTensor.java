package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * Expected source tensor and runtime role used by the T5 layout validator.
 */
final class T5ExpectedTensor {
    private final String role;
    private final String sourceName;
    private final String runtimeName;
    private final boolean required;
    private final long[] expectedDims;

    T5ExpectedTensor(String role, String sourceName, String runtimeName, boolean required, long... expectedDims) {
        this.role = Objects.requireNonNull(role, "role");
        this.sourceName = Objects.requireNonNull(sourceName, "sourceName");
        this.runtimeName = Objects.requireNonNull(runtimeName, "runtimeName");
        this.required = required;
        this.expectedDims = expectedDims == null ? new long[0] : expectedDims.clone();
    }

    String role() {
        return role;
    }

    String sourceName() {
        return sourceName;
    }

    String runtimeName() {
        return runtimeName;
    }

    boolean required() {
        return required;
    }

    long[] expectedDims() {
        return expectedDims.clone();
    }
}
