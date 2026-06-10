package com.aresstack.windirectml.inference.decoderonly;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.stream.IntStream;

/**
 * Shared dense math hot path for decoder-only reference runtimes.
 *
 * <p>This class intentionally lives in {@code decoderonly} instead of a concrete model package. Qwen, SmolLM2
 * and later decoder-only families can reuse the same SIMD/scalar reference implementation while native WARP
 * kernels are introduced behind the same projection boundary.</p>
 */
public final class DecoderOnlyReferenceDenseOps {

    private static final boolean ENABLED;
    private static final VectorSpecies<Float> SPECIES;
    private static final int LANES;
    private static final boolean PARALLEL_DENSE;
    private static final int PARALLEL_ROW_THRESHOLD;
    private static final int PARALLEL_COLUMN_THRESHOLD;

    static {
        boolean enabled;
        VectorSpecies<Float> species = null;
        int lanes = 0;
        try {
            species = FloatVector.SPECIES_PREFERRED;
            lanes = species.length();
            enabled = lanes >= 4;
        } catch (Throwable ignored) {
            enabled = false;
        }
        ENABLED = enabled;
        SPECIES = species;
        LANES = lanes;
        PARALLEL_DENSE = Boolean.parseBoolean(System.getProperty(
                "windirectml.decoderonly.reference.parallelDense",
                System.getProperty("windirectml.smollm2.reference.parallelDense", "true")));
        PARALLEL_ROW_THRESHOLD = Integer.getInteger(
                "windirectml.decoderonly.reference.parallelRowsThreshold",
                Integer.getInteger("windirectml.smollm2.reference.parallelRowsThreshold", 1024));
        PARALLEL_COLUMN_THRESHOLD = Integer.getInteger(
                "windirectml.decoderonly.reference.parallelColumnsThreshold",
                Integer.getInteger("windirectml.smollm2.reference.parallelColumnsThreshold", 256));
    }

    private DecoderOnlyReferenceDenseOps() {
    }

    public static boolean enabled() {
        return ENABLED;
    }

    public static boolean parallelEnabled() {
        return PARALLEL_DENSE;
    }

    public static float dot(float[] left, int leftOffset, float[] right, int rightOffset, int length) {
        requireRange(left, leftOffset, length, "left");
        requireRange(right, rightOffset, length, "right");
        if (!ENABLED) {
            return scalarDot(left, leftOffset, right, rightOffset, length);
        }

        FloatVector sumVector = FloatVector.zero(SPECIES);
        int upperBound = SPECIES.loopBound(length);
        int index = 0;
        for (; index < upperBound; index += LANES) {
            FloatVector leftVector = FloatVector.fromArray(SPECIES, left, leftOffset + index);
            FloatVector rightVector = FloatVector.fromArray(SPECIES, right, rightOffset + index);
            sumVector = leftVector.fma(rightVector, sumVector);
        }

        float sum = sumVector.reduceLanes(VectorOperators.ADD);
        for (; index < length; index++) {
            sum += left[leftOffset + index] * right[rightOffset + index];
        }
        return sum;
    }

    public static float[] multiplyRows(float[] matrix, int rows, int cols, float[] input) {
        float[] output = new float[rows];
        multiplyRows(matrix, rows, cols, input, output);
        return output;
    }

    public static void multiplyRows(float[] matrix, int rows, int cols, float[] input, float[] output) {
        requireMatrix(matrix, rows, cols);
        if (input.length != cols) {
            throw new IllegalArgumentException("Input width mismatch: expected " + cols + " but got " + input.length);
        }
        if (output.length < rows) {
            throw new IllegalArgumentException("Output height mismatch: expected at least " + rows
                    + " but got " + output.length);
        }
        if (shouldUseParallelRows(rows, cols)) {
            IntStream.range(0, rows)
                    .parallel()
                    .forEach(row -> output[row] = dot(matrix, row * cols, input, 0, cols));
            return;
        }
        for (int row = 0; row < rows; row++) {
            output[row] = dot(matrix, row * cols, input, 0, cols);
        }
    }

    public static void gatedSiluMultiply(float[] gate, float[] up) {
        if (gate.length != up.length) {
            throw new IllegalArgumentException("gate and up vectors must have the same length");
        }
        for (int i = 0; i < gate.length; i++) {
            gate[i] = silu(gate[i]) * up[i];
        }
    }

    public static float silu(float value) {
        return value / (1.0f + (float) Math.exp(-value));
    }

    private static boolean shouldUseParallelRows(int rows, int cols) {
        return PARALLEL_DENSE
                && rows >= PARALLEL_ROW_THRESHOLD
                && cols >= PARALLEL_COLUMN_THRESHOLD
                && Runtime.getRuntime().availableProcessors() > 1;
    }

    private static float scalarDot(float[] left, int leftOffset, float[] right, int rightOffset, int length) {
        float sum = 0.0f;
        for (int index = 0; index < length; index++) {
            sum += left[leftOffset + index] * right[rightOffset + index];
        }
        return sum;
    }

    private static void requireMatrix(float[] matrix, int rows, int cols) {
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("rows and cols must be non-negative");
        }
        long expected = (long) rows * cols;
        if (matrix.length < expected) {
            throw new IllegalArgumentException("Matrix too small: " + matrix.length + " < " + expected);
        }
    }

    private static void requireRange(float[] values, int offset, int length, String name) {
        if (offset < 0 || length < 0 || (long) offset + length > values.length) {
            throw new IllegalArgumentException(name + " range out of bounds: offset=" + offset
                    + ", length=" + length + ", size=" + values.length);
        }
    }
}
