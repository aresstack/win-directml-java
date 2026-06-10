package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.windows.MatMulNBitsKernel;
import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.Objects;

/**
 * DirectML/WARP-backed dense projection shared by decoder-only model families.
 *
 * <p>This is the first reusable native execution seam for Qwen/SmolLM2-style decoder-only models. It uses the
 * existing DirectML matvec/GEMM kernel and exposes a family-neutral projection API. Concrete model runtimes should
 * compose this class instead of introducing per-family WARP dense adapters.</p>
 */
public final class DecoderOnlyWarpDenseProjection implements DecoderOnlyDenseProjection {
    private final String name;
    private final int inputSize;
    private final int outputSize;
    private final MatMulNBitsKernel kernel;
    private boolean batchDisabled;
    private boolean closed;

    private DecoderOnlyWarpDenseProjection(String name, int inputSize, int outputSize, MatMulNBitsKernel kernel) {
        this.name = Objects.requireNonNull(name, "name");
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.kernel = Objects.requireNonNull(kernel, "kernel");
    }

    public static DecoderOnlyWarpDenseProjection fromRowMajorWeights(WindowsBindings windowsBindings,
                                                                      String name,
                                                                      int outputSize,
                                                                      int inputSize,
                                                                      float[] weights) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        validateShape(outputSize, inputSize, weights);
        MatMulNBitsKernel kernel = MatMulNBitsKernel.fromDequantizedWeights(
                windowsBindings, outputSize, inputSize, weights);
        return new DecoderOnlyWarpDenseProjection(name, inputSize, outputSize, kernel);
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public int inputSize() {
        return inputSize;
    }

    @Override
    public int outputSize() {
        return outputSize;
    }

    @Override
    public float[] project(float[] input) {
        float[] output = new float[outputSize];
        projectInto(input, output);
        return output;
    }

    @Override
    public void projectInto(float[] input, float[] output) {
        ensureOpen();
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input length mismatch for " + name
                    + ": input=" + input.length + ", expected=" + inputSize);
        }
        if (output.length < outputSize) {
            throw new IllegalArgumentException("Output buffer too small for " + name
                    + ": output=" + output.length + ", expected at least=" + outputSize);
        }
        kernel.matvec(input, output);
    }

    @Override
    public void projectSequenceInto(float[] input, int sequenceLength, float[] output) {
        ensureOpen();
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        if (sequenceLength < 1) {
            throw new IllegalArgumentException("sequenceLength must be positive: " + sequenceLength);
        }
        if (input.length != sequenceLength * inputSize) {
            throw new IllegalArgumentException("Sequence input length mismatch for " + name
                    + ": values=" + input.length + ", expected=" + (sequenceLength * inputSize));
        }
        if (output.length < sequenceLength * outputSize) {
            throw new IllegalArgumentException("Sequence output length mismatch for " + name
                    + ": values=" + output.length + ", expected at least=" + (sequenceLength * outputSize));
        }
        if (sequenceLength == 1 || batchDisabled || !kernel.supportsBatch()) {
            DecoderOnlyDenseProjection.super.projectSequenceInto(input, sequenceLength, output);
            return;
        }
        try {
            kernel.matmulBatch(input, output, sequenceLength);
        } catch (RuntimeException ex) {
            batchDisabled = true;
            DecoderOnlyDenseProjection.super.projectSequenceInto(input, sequenceLength, output);
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            kernel.close();
        }
    }

    public boolean isClosed() {
        return closed;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Decoder-only WARP projection is closed: " + name);
        }
    }

    private static void validateShape(int outputSize, int inputSize, float[] weights) {
        Objects.requireNonNull(weights, "weights");
        if (outputSize < 1) {
            throw new IllegalArgumentException("outputSize must be positive: " + outputSize);
        }
        if (inputSize < 1) {
            throw new IllegalArgumentException("inputSize must be positive: " + inputSize);
        }
        long expected = (long) outputSize * inputSize;
        if (weights.length != expected) {
            throw new IllegalArgumentException("Weight length mismatch for decoder-only WARP projection: weights="
                    + weights.length + ", expected=" + expected);
        }
    }
}
