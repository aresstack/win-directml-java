package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.Objects;

/**
 * WARP/DirectML encoder execution boundary for the T5 family.
 *
 * <p>v40 introduces this boundary before moving individual encoder operators to
 * native execution. The class intentionally keeps the T5 encoder flow separate
 * from decoder-only infrastructure and delegates to the reference encoder until
 * WARP kernels are attached behind the same contract.</p>
 */
public final class T5WarpEncoderPipeline implements T5EncoderRunner, AutoCloseable {
    private final WindowsBindings windowsBindings;
    private final T5EncoderPipeline referenceEncoder;
    private boolean closed;

    private T5WarpEncoderPipeline(WindowsBindings windowsBindings, T5EncoderPipeline referenceEncoder) {
        this.windowsBindings = Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.referenceEncoder = Objects.requireNonNull(referenceEncoder, "referenceEncoder");
    }

    /**
     * Create a T5 encoder runner whose public boundary is ready for WARP-backed
     * execution while encoder math still uses the correctness path.
     *
     * @param windowsBindings initialized Windows bindings using WARP/AUTO
     * @param weights         payload-backed T5 weights
     * @return encoder execution boundary for T5
     */
    public static T5WarpEncoderPipeline from(WindowsBindings windowsBindings, T5Weights weights) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        Objects.requireNonNull(weights, "weights");
        return new T5WarpEncoderPipeline(windowsBindings, T5EncoderPipeline.from(weights));
    }

    @Override
    public T5EncoderOutput encode(int[] inputTokenIds) {
        ensureOpen();
        return referenceEncoder.encode(inputTokenIds);
    }

    @Override
    public T5EncoderOutput encode(int[] inputTokenIds, boolean[] attentionMask) {
        ensureOpen();
        return referenceEncoder.encode(inputTokenIds, attentionMask);
    }

    @Override
    public String executionMode() {
        return "warp-encoder-boundary+reference-encoder";
    }

    public WindowsBindings windowsBindings() {
        return windowsBindings;
    }

    public T5EncoderPipeline referenceEncoder() {
        return referenceEncoder;
    }

    public boolean isClosed() {
        return closed;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("T5 WARP encoder pipeline is closed");
        }
    }

    @Override
    public void close() {
        closed = true;
    }
}
