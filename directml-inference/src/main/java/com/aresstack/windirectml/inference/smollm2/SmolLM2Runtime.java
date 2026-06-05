package com.aresstack.windirectml.inference.smollm2;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Public SmolLM2 runtime facade. Text generation waits for tokenizer integration; token-level reference generation is available.
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
        ensureOpen();
        throw new SmolLM2RuntimeUnsupportedException(UNSUPPORTED_MESSAGE);
    }

    /**
     * Generate token IDs through the correctness-first reference pipeline.
     *
     * <p>This method deliberately accepts token IDs instead of text because SmolLM2 tokenizer integration is a
     * separate production work item.</p>
     */
    public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        SmolLM2Weights weights = runtimePackage.requireWeights();
        return new SmolLM2ReferenceGenerationLoop(weights).generate(request);
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("SmolLM2 runtime is closed");
        }
    }

    public SmolLM2RuntimePackage runtimePackage() {
        return runtimePackage;
    }

    @Override
    public void close() {
        closed.set(true);
    }
}
