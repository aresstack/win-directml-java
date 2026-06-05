package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.Objects;

/**
 * WARP/DirectML boundary for the T5 decoder stage.
 *
 * <p>This class routes decoder linear projections through WARP/DirectML while
 * keeping the seq2seq decoder flow independent from decoder-only runtime
 * infrastructure. Softmax, residual connections, and cache structure remain in
 * the T5 reference path until dedicated native kernels are added.</p>
 */
public final class T5WarpDecoderPipeline implements T5DecoderRunner, AutoCloseable {
    private final WindowsBindings windowsBindings;
    private final T5Weights weights;
    private final T5DecoderPipeline warpPipeline;
    private boolean closed;

    private T5WarpDecoderPipeline(WindowsBindings windowsBindings, T5Weights weights,
                                  T5DecoderPipeline warpPipeline) {
        this.windowsBindings = Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.warpPipeline = Objects.requireNonNull(warpPipeline, "warpPipeline");
    }

    /**
     * Create the WARP decoder boundary.
     *
     * @param windowsBindings initialized Windows bindings using WARP/AUTO
     * @param weights         loaded T5 weights
     * @return decoder boundary instance
     */
    public static T5WarpDecoderPipeline from(WindowsBindings windowsBindings, T5Weights weights) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        Objects.requireNonNull(weights, "weights");
        T5WarpLinearProjectionFactory projectionFactory = new T5WarpLinearProjectionFactory(windowsBindings);
        return new T5WarpDecoderPipeline(windowsBindings, weights, T5DecoderPipeline.from(weights, projectionFactory));
    }

    @Override
    public String executionMode() {
        return "warp-decoder-linear-projections";
    }

    @Override
    public T5DecoderState decode(int[] decoderInputIds, T5EncoderOutput encoderOutput) {
        ensureOpen();
        return warpPipeline.decode(decoderInputIds, encoderOutput);
    }

    @Override
    public T5DecoderState decodeStep(int decoderTokenId, T5EncoderOutput encoderOutput, T5DecoderCache cache) {
        ensureOpen();
        return warpPipeline.decodeStep(decoderTokenId, encoderOutput, cache);
    }

    /**
     * Return the reference pipeline currently used behind the boundary.
     *
     * @return reference decoder pipeline
     */
    public T5DecoderPipeline warpPipeline() {
        return warpPipeline;
    }

    /**
     * Return the loaded weights associated with this decoder boundary.
     *
     * @return T5 weights
     */
    public T5Weights weights() {
        return weights;
    }

    /**
     * Return the native binding handle holder associated with this boundary.
     *
     * @return Windows bindings
     */
    public WindowsBindings windowsBindings() {
        return windowsBindings;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("T5 WARP decoder pipeline is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            warpPipeline.close();
        }
    }
}
