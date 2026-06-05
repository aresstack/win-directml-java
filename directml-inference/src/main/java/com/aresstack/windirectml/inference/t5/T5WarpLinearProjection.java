package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.windows.MatMulNBitsKernel;
import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.Objects;

/**
 * WARP/DirectML-backed T5 linear projection.
 *
 * <p>This adapter reuses the existing DirectML matvec kernel and keeps T5
 * encoder/decoder math independent from Qwen or decoder-only runtimes. It is a
 * conservative first WARP step: each projection is uploaded once and then used
 * for token-wise matvec calls.</p>
 */
public final class T5WarpLinearProjection implements T5LinearProjection {
    private final String name;
    private final int inputSize;
    private final int outputSize;
    private final MatMulNBitsKernel kernel;
    private boolean batchDisabled;
    private boolean closed;

    private T5WarpLinearProjection(String name, int inputSize, int outputSize, MatMulNBitsKernel kernel) {
        this.name = Objects.requireNonNull(name, "name");
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.kernel = Objects.requireNonNull(kernel, "kernel");
    }

    public static T5WarpLinearProjection from(WindowsBindings windowsBindings, T5TensorData weight) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        Objects.requireNonNull(weight, "weight");
        if (weight.rank() != 2) {
            throw new IllegalArgumentException("WARP projection weight must be rank 2: " + weight.name());
        }
        int outputSize = weight.dim(0);
        int inputSize = weight.dim(1);
        MatMulNBitsKernel kernel = MatMulNBitsKernel.fromDequantizedWeights(
                windowsBindings, outputSize, inputSize, weight.values());
        return new T5WarpLinearProjection(weight.name(), inputSize, outputSize, kernel);
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
    public float[] apply(float[] input) {
        ensureOpen();
        Objects.requireNonNull(input, "input");
        if (input.length != inputSize) {
            throw new IllegalArgumentException("WARP projection input length mismatch for " + name
                    + ": input=" + input.length + ", expected=" + inputSize);
        }
        return kernel.matvec(input);
    }


    @Override
    public float[] applySequence(float[] input, int sequenceLength, int inputSize) {
        ensureOpen();
        Objects.requireNonNull(input, "input");
        if (inputSize != inputSize()) {
            throw new IllegalArgumentException("WARP projection input size mismatch for " + name
                    + ": input=" + inputSize + ", expected=" + inputSize());
        }
        if (sequenceLength < 1) {
            throw new IllegalArgumentException("sequenceLength must be positive for " + name + ": " + sequenceLength);
        }
        if (input.length != sequenceLength * inputSize) {
            throw new IllegalArgumentException("WARP projection sequence length mismatch for " + name
                    + ": values=" + input.length + ", expected=" + (sequenceLength * inputSize));
        }
        float[] result = new float[sequenceLength * outputSize];
        if (sequenceLength == 1 || batchDisabled || !kernel.supportsBatch()) {
            applyRows(input, result, sequenceLength);
            return result;
        }
        try {
            kernel.matmulBatch(input, result, sequenceLength);
            return result;
        } catch (RuntimeException ex) {
            batchDisabled = true;
            applyRows(input, result, sequenceLength);
            return result;
        }
    }

    private void applyRows(float[] input, float[] result, int sequenceLength) {
        float[] row = new float[inputSize];
        float[] rowOutput = new float[outputSize];
        for (int token = 0; token < sequenceLength; token++) {
            System.arraycopy(input, token * inputSize, row, 0, inputSize);
            kernel.matvec(row, rowOutput);
            System.arraycopy(rowOutput, 0, result, token * outputSize, outputSize);
        }
    }

    public boolean isClosed() {
        return closed;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("T5 WARP projection is closed: " + name);
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
