package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.Objects;

/**
 * WARP/DirectML encoder execution boundary for the T5 family.
 *
 * <p>This class keeps the T5 encoder flow separate from decoder-only infrastructure
 * and routes encoder linear projections through WARP/DirectML. Non-linear
 * operations such as softmax, residual addition, and layer normalization still
 * use the T5 reference implementation until dedicated native kernels are added.</p>
 */
public final class T5WarpEncoderPipeline implements T5EncoderRunner, AutoCloseable {
    private final WindowsBindings windowsBindings;
    private final T5EncoderPipeline warpEncoder;
    private boolean closed;

    private T5WarpEncoderPipeline(WindowsBindings windowsBindings, T5EncoderPipeline warpEncoder) {
        this.windowsBindings = Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.warpEncoder = Objects.requireNonNull(warpEncoder, "warpEncoder");
    }

    /**
     * Create a T5 encoder runner whose public boundary is ready for WARP-backed
     * execution for encoder linear projections.
     *
     * @param windowsBindings initialized Windows bindings using WARP/AUTO
     * @param weights         payload-backed T5 weights
     * @return encoder execution boundary for T5
     */
    public static T5WarpEncoderPipeline from(WindowsBindings windowsBindings, T5Weights weights) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        Objects.requireNonNull(weights, "weights");
        T5WarpLinearProjectionFactory projectionFactory = new T5WarpLinearProjectionFactory(windowsBindings);
        return new T5WarpEncoderPipeline(windowsBindings, T5EncoderPipeline.from(weights, projectionFactory));
    }

    @Override
    public T5EncoderOutput encode(int[] inputTokenIds) {
        ensureOpen();
        return warpEncoder.encode(inputTokenIds);
    }

    @Override
    public T5EncoderOutput encode(int[] inputTokenIds, boolean[] attentionMask) {
        ensureOpen();
        return warpEncoder.encode(inputTokenIds, attentionMask);
    }

    @Override
    public String executionMode() {
        return "warp-encoder-linear-projections";
    }

    public WindowsBindings windowsBindings() {
        return windowsBindings;
    }

    public T5EncoderPipeline warpEncoder() {
        return warpEncoder;
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
        if (!closed) {
            closed = true;
            warpEncoder.close();
        }
    }
}
