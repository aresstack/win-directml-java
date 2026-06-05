package com.aresstack.windirectml.inference.t5;

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
    private boolean closed;

    private T5Runtime(T5RuntimePackage runtimePackage, T5Weights weights) {
        this.runtimePackage = Objects.requireNonNull(runtimePackage, "runtimePackage");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.encoderPipeline = T5EncoderPipeline.from(weights);
        this.decoderPipeline = T5DecoderPipeline.from(weights);
    }

    public static T5Runtime load(T5RuntimePackage runtimePackage) throws java.io.IOException {
        Objects.requireNonNull(runtimePackage, "runtimePackage");
        return new T5Runtime(runtimePackage, runtimePackage.weights());
    }

    public T5RuntimeResult generate(T5RuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        throw new T5UnsupportedRuntimeException(UNSUPPORTED_MESSAGE);
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

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("T5 runtime is closed");
        }
    }

    @Override
    public void close() {
        closed = true;
    }
}
