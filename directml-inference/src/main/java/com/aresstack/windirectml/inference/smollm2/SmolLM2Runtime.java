package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Public SmolLM2 runtime shell. It validates package metadata but does not execute yet.
 */
public final class SmolLM2Runtime implements AutoCloseable {

    private static final String UNSUPPORTED_MESSAGE =
            "SmolLM2 runtime execution is not implemented yet. Compile/import support is available, "
                    + "but runtimeLoadable=false packages cannot generate text.";

    private final SmolLM2RuntimePackage runtimePackage;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private SmolLM2Runtime(SmolLM2RuntimePackage runtimePackage) {
        this.runtimePackage = Objects.requireNonNull(runtimePackage, "runtimePackage");
    }

    public static SmolLM2Runtime load(SmolLM2RuntimePackage runtimePackage) {
        return new SmolLM2Runtime(runtimePackage);
    }

    public SmolLM2RuntimeResult generate(SmolLM2RuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        if (closed.get()) {
            throw new IllegalStateException("SmolLM2 runtime is closed");
        }
        throw new SmolLM2RuntimeUnsupportedException(UNSUPPORTED_MESSAGE);
    }

    public SmolLM2RuntimePackage runtimePackage() {
        return runtimePackage;
    }

    @Override
    public void close() {
        closed.set(true);
    }
}
