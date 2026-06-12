package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.MatMulNBitsKernel;
import com.aresstack.windirectml.windows.WindowsBindings;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * WARP/DirectML-backed LM-head projector for T5 generation.
 *
 * <p>This is the first native execution boundary for the T5 family. Encoder and
 * decoder blocks can still use the reference path while the expensive vocabulary
 * projection already runs through the existing DirectML matvec kernel. Later T5
 * patches should replace the remaining reference blocks with encoder/decoder
 * WARP pipelines and keep this class as the LM-head projection adapter.</p>
 */
public final class T5WarpLmHead implements T5LogitProjector, AutoCloseable {
    private final MatMulNBitsKernel kernel;
    private final int hiddenSize;
    private final int vocabularySize;
    private boolean closed;

    private T5WarpLmHead(MatMulNBitsKernel kernel, int hiddenSize, int vocabularySize) {
        this.kernel = Objects.requireNonNull(kernel, "kernel");
        this.hiddenSize = hiddenSize;
        this.vocabularySize = vocabularySize;
    }

    /**
     * Create a WARP/DirectML LM-head projector from payload-backed T5 weights.
     *
     * <p>The current bridge materializes the dense LM-head tensor once to build
     * the existing DirectML kernel. This is acceptable for the v39 bridge and
     * keeps generation independent from Java dense multiplication. A later
     * package-loader patch can upload the runtime tensor payload directly.</p>
     *
     * @param windowsBindings initialized Windows bindings using WARP/AUTO
     * @param weights         payload-backed T5 weights
     * @return WARP-backed LM-head projector
     */
    public static T5WarpLmHead from(WindowsBindings windowsBindings, T5Weights weights) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        Objects.requireNonNull(weights, "weights");
        T5TensorData lmHead = T5TensorData.from(weights.lmHead());
        if (lmHead.rank() != 2) {
            throw new IllegalArgumentException("T5 LM head must be rank 2: " + lmHead.name());
        }
        int vocabularySize = lmHead.dim(0);
        int hiddenSize = lmHead.dim(1);
        // Heap-light FP32 path: upload the mmap ByteBuffer slice of the (largest) LM-head matrix directly,
        // skipping the host float[]. FLOAT16 LM heads fall back to the float[] path.
        ByteBuffer fp32 = lmHead.fp32LittleEndianSource();
        MatMulNBitsKernel kernel = (fp32 != null)
                ? MatMulNBitsKernel.fromDequantizedWeights(windowsBindings, vocabularySize, hiddenSize, fp32)
                : MatMulNBitsKernel.fromDequantizedWeights(windowsBindings, vocabularySize, hiddenSize, lmHead.values());
        return new T5WarpLmHead(kernel, hiddenSize, vocabularySize);
    }

    @Override
    public float[] logits(float[] decoderHiddenState) {
        float[] logits = new float[vocabularySize];
        logitsInto(decoderHiddenState, logits);
        return logits;
    }

    @Override
    public void logitsInto(float[] decoderHiddenState, float[] outputLogits) {
        ensureOpen();
        Objects.requireNonNull(decoderHiddenState, "decoderHiddenState");
        Objects.requireNonNull(outputLogits, "outputLogits");
        if (decoderHiddenState.length != hiddenSize) {
            throw new IllegalArgumentException("Decoder hidden state length mismatch for T5 WARP LM head: hidden="
                    + decoderHiddenState.length + ", expected=" + hiddenSize);
        }
        if (outputLogits.length < vocabularySize) {
            throw new IllegalArgumentException("T5 WARP LM head output buffer too small: " + outputLogits.length
                    + " < " + vocabularySize);
        }
        kernel.matvec(decoderHiddenState, outputLogits);
    }

    @Override
    public int vocabularySize() {
        return vocabularySize;
    }

    public boolean isClosed() {
        return closed;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("T5 WARP LM head is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            kernel.close();
        }
    }
}
