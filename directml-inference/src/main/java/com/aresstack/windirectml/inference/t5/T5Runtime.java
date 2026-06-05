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
    private final T5DecoderPipeline decoderPipeline;
    private final T5GenerationLoop generationLoop;
    private final T5LogitProjector logitProjector;
    private final String executionMode;
    private boolean closed;

    private T5Runtime(T5RuntimePackage runtimePackage, T5Weights weights,
                      T5LogitProjector logitProjector, String executionMode) {
        this.runtimePackage = Objects.requireNonNull(runtimePackage, "runtimePackage");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.logitProjector = Objects.requireNonNull(logitProjector, "logitProjector");
        this.executionMode = Objects.requireNonNull(executionMode, "executionMode");
        this.encoderPipeline = T5EncoderPipeline.from(weights);
        this.decoderPipeline = T5DecoderPipeline.from(weights);
        this.generationLoop = T5GenerationLoop.greedy(encoderPipeline, decoderPipeline, logitProjector);
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
        return encoderPipeline.encode(inputTokenIds);
    }

    /**
     * Execute the T5 encoder reference pipeline with an explicit attention mask.
     */
    public T5EncoderOutput encode(int[] inputTokenIds, boolean[] attentionMask) {
        ensureOpen();
        return encoderPipeline.encode(inputTokenIds, attentionMask);
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
            if (logitProjector instanceof AutoCloseable) {
                try {
                    ((AutoCloseable) logitProjector).close();
                } catch (Exception e) {
                    throw new RuntimeException("Failed to close T5 logit projector", e);
                }
            }
        }
    }
}
