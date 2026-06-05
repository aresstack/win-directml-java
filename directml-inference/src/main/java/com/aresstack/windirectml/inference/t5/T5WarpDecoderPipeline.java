package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.Objects;

/**
 * WARP/DirectML boundary for the T5 decoder stage.
 *
 * <p>v41 keeps the decoder math on the reference implementation behind this
 * boundary. The class exists so decoder self-attention, cross-attention, and
 * the future decoder KV/cache layout can be replaced by native WARP operators
 * without changing {@link T5GenerationLoop} or reusing decoder-only runtime
 * infrastructure.</p>
 */
public final class T5WarpDecoderPipeline implements T5DecoderRunner, AutoCloseable {
    private final WindowsBindings windowsBindings;
    private final T5Weights weights;
    private final T5DecoderPipeline referencePipeline;
    private boolean closed;

    private T5WarpDecoderPipeline(WindowsBindings windowsBindings, T5Weights weights,
                                  T5DecoderPipeline referencePipeline) {
        this.windowsBindings = Objects.requireNonNull(windowsBindings, "windowsBindings");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.referencePipeline = Objects.requireNonNull(referencePipeline, "referencePipeline");
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
        return new T5WarpDecoderPipeline(windowsBindings, weights, T5DecoderPipeline.from(weights));
    }

    @Override
    public String executionMode() {
        return "warp-decoder-boundary";
    }

    @Override
    public T5DecoderState decode(int[] decoderInputIds, T5EncoderOutput encoderOutput) {
        ensureOpen();
        return referencePipeline.decode(decoderInputIds, encoderOutput);
    }

    @Override
    public T5DecoderState decodeStep(int decoderTokenId, T5EncoderOutput encoderOutput, T5DecoderCache cache) {
        ensureOpen();
        return referencePipeline.decodeStep(decoderTokenId, encoderOutput, cache);
    }

    /**
     * Return the reference pipeline currently used behind the boundary.
     *
     * @return reference decoder pipeline
     */
    public T5DecoderPipeline referencePipeline() {
        return referencePipeline;
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
        closed = true;
    }
}
