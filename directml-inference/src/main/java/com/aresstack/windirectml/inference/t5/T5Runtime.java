package com.aresstack.windirectml.inference.t5;

import java.util.Objects;

/**
 * API shell for the future T5 WARP runtime.
 */
public final class T5Runtime implements AutoCloseable {
    public static final String UNSUPPORTED_MESSAGE = "T5 WARP runtime is not implemented yet.";

    private final T5RuntimePackage runtimePackage;
    private boolean closed;

    private T5Runtime(T5RuntimePackage runtimePackage) {
        this.runtimePackage = Objects.requireNonNull(runtimePackage, "runtimePackage");
    }

    public static T5Runtime load(T5RuntimePackage runtimePackage) {
        return new T5Runtime(runtimePackage);
    }

    public T5RuntimeResult generate(T5RuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        if (closed) {
            throw new IllegalStateException("T5 runtime is closed");
        }
        throw new T5UnsupportedRuntimeException(UNSUPPORTED_MESSAGE);
    }

    public T5RuntimePackage runtimePackage() {
        return runtimePackage;
    }

    @Override
    public void close() {
        closed = true;
    }
}
