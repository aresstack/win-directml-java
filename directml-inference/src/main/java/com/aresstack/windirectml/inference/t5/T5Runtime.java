package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.Objects;

/**
 * API shell for the future T5 WARP runtime.
 */
public final class T5Runtime implements AutoCloseable {
    public static final String UNSUPPORTED_MESSAGE = "T5 WARP runtime is not implemented yet.";

    private final T5RuntimePackage runtimePackage;
    private final T5Weights weights;
    private final T5EncoderPipeline encoderPipeline;
    private final T5EncoderRunner encoderRunner;
    private final T5DecoderPipeline decoderPipeline;
    private final T5GenerationLoop generationLoop;
    private final T5LogitProjector logitProjector;
    private final String executionMode;
    private boolean closed;

    private T5Runtime(T5RuntimePackage runtimePackage, T5Weights weights,
                      T5LogitProjector logitProjector, String executionMode) {
        this(runtimePackage, weights, T5EncoderPipeline.from(weights), logitProjector, executionMode);
    }

    private T5Runtime(T5RuntimePackage runtimePackage, T5Weights weights,
                      T5EncoderRunner encoderRunner, T5LogitProjector logitProjector,
                      String executionMode) {
        this.runtimePackage = Objects.requireNonNull(runtimePackage, "runtimePackage");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.encoderRunner = Objects.requireNonNull(encoderRunner, "encoderRunner");
        this.logitProjector = Objects.requireNonNull(logitProjector, "logitProjector");
        this.executionMode = Objects.requireNonNull(executionMode, "executionMode");
        this.encoderPipeline = encoderRunner instanceof T5EncoderPipeline
                ? (T5EncoderPipeline) encoderRunner
                : T5EncoderPipeline.from(weights);
        this.decoderPipeline = T5DecoderPipeline.from(weights);
        this.generationLoop = T5GenerationLoop.greedy(encoderRunner, decoderPipeline, logitProjector);
    }

    public static T5Runtime load(T5RuntimePackage runtimePackage) throws java.io.IOException {
        Objects.requireNonNull(runtimePackage, "runtimePackage");
        T5Weights weights = runtimePackage.weights();
        return new T5Runtime(runtimePackage, weights, T5LmHead.from(weights), "reference");
    }

    /**
     * Load a T5 runtime whose LM-head projection is backed by WARP/DirectML.
     *
     * <p>Encoder and decoder execution are still the reference pipelines in v39.
     * This method exists to move the expensive vocabulary projection behind the
     * same native execution boundary used by Qwen while keeping the T5 seq2seq
     * control flow independent from decoder-only code.</p>
     *
     * @param runtimePackage  T5 runtime package
     * @param windowsBindings initialized Windows bindings using WARP/AUTO
     * @return runtime using WARP for LM-head projection
     * @throws java.io.IOException if package weights cannot be loaded
     */
    public static T5Runtime loadWarpLmHead(T5RuntimePackage runtimePackage,
                                           WindowsBindings windowsBindings) throws java.io.IOException {
        Objects.requireNonNull(runtimePackage, "runtimePackage");
        T5Weights weights = runtimePackage.weights();
        return new T5Runtime(runtimePackage, weights, T5WarpLmHead.from(windowsBindings, weights),
                "reference+warp-lm-head");
    }

    /**
     * Load a T5 runtime whose encoder stage is routed through the WARP boundary.
     *
     * <p>v40 keeps encoder math on the reference implementation behind this
     * boundary. Future patches can attach native encoder operators without
     * changing the seq2seq runtime API or reusing decoder-only infrastructure.</p>
     *
     * @param runtimePackage  T5 runtime package
     * @param windowsBindings initialized Windows bindings using WARP/AUTO
     * @return runtime using the WARP encoder boundary
     * @throws java.io.IOException if package weights cannot be loaded
     */
    public static T5Runtime loadWarpEncoder(T5RuntimePackage runtimePackage,
                                            WindowsBindings windowsBindings) throws java.io.IOException {
        Objects.requireNonNull(runtimePackage, "runtimePackage");
        T5Weights weights = runtimePackage.weights();
        T5EncoderRunner encoderRunner = T5WarpEncoderPipeline.from(windowsBindings, weights);
        return new T5Runtime(runtimePackage, weights, encoderRunner, T5LmHead.from(weights),
                "warp-encoder-boundary+reference-decoder+reference-lm-head");
    }

    /**
     * Load a T5 runtime with both the encoder boundary and the LM-head projector
     * routed through WARP/DirectML contracts.
     *
     * @param runtimePackage  T5 runtime package
     * @param windowsBindings initialized Windows bindings using WARP/AUTO
     * @return runtime using WARP boundaries for encoder and LM-head stages
     * @throws java.io.IOException if package weights cannot be loaded
     */
    public static T5Runtime loadWarpEncoderAndLmHead(T5RuntimePackage runtimePackage,
                                                     WindowsBindings windowsBindings) throws java.io.IOException {
        Objects.requireNonNull(runtimePackage, "runtimePackage");
        T5Weights weights = runtimePackage.weights();
        T5EncoderRunner encoderRunner = T5WarpEncoderPipeline.from(windowsBindings, weights);
        return new T5Runtime(runtimePackage, weights, encoderRunner, T5WarpLmHead.from(windowsBindings, weights),
                "warp-encoder-boundary+reference-decoder+warp-lm-head");
    }

    public T5RuntimeResult generate(T5RuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        return generationLoop.generate(request);
    }

    /**
     * Execute the T5 encoder reference pipeline.
     *
     * <p>This is a correctness path for v36. It deliberately does not imply
     * that decoder/generation runtime is loadable yet.</p>
     */
    public T5EncoderOutput encode(int[] inputTokenIds) {
        ensureOpen();
        return encoderRunner.encode(inputTokenIds);
    }

    /**
     * Execute the T5 encoder reference pipeline with an explicit attention mask.
     */
    public T5EncoderOutput encode(int[] inputTokenIds, boolean[] attentionMask) {
        ensureOpen();
        return encoderRunner.encode(inputTokenIds, attentionMask);
    }

    /**
     * Execute the T5 decoder reference pipeline over the supplied decoder prefix.
     */
    public T5DecoderState decode(int[] decoderInputIds, T5EncoderOutput encoderOutput) {
        ensureOpen();
        return decoderPipeline.decode(decoderInputIds, encoderOutput);
    }

    /**
     * Execute one T5 decoder reference step using the current decoder cache boundary.
     */
    public T5DecoderState decodeStep(int decoderTokenId, T5EncoderOutput encoderOutput, T5DecoderCache cache) {
        ensureOpen();
        return decoderPipeline.decodeStep(decoderTokenId, encoderOutput, cache);
    }

    public T5RuntimePackage runtimePackage() {
        return runtimePackage;
    }

    public T5Weights weights() {
        return weights;
    }

    public T5EncoderPipeline encoderPipeline() {
        return encoderPipeline;
    }

    public T5EncoderRunner encoderRunner() {
        return encoderRunner;
    }

    public T5DecoderPipeline decoderPipeline() {
        return decoderPipeline;
    }

    public T5GenerationLoop generationLoop() {
        return generationLoop;
    }

    public String executionMode() {
        return executionMode;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("T5 runtime is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            RuntimeException failure = null;
            if (logitProjector instanceof AutoCloseable) {
                try {
                    ((AutoCloseable) logitProjector).close();
                } catch (Exception e) {
                    failure = new RuntimeException("Failed to close T5 logit projector", e);
                }
            }
            if (encoderRunner instanceof AutoCloseable) {
                try {
                    ((AutoCloseable) encoderRunner).close();
                } catch (Exception e) {
                    RuntimeException encoderFailure = new RuntimeException("Failed to close T5 encoder runner", e);
                    if (failure == null) {
                        failure = encoderFailure;
                    } else {
                        failure.addSuppressed(encoderFailure);
                    }
                }
            }
            if (failure != null) {
                throw failure;
            }
        }
    }
}
