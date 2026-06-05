package com.aresstack.windirectml.inference.smollm2;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.stream.IntStream;

/**
 * SIMD helpers for the SmolLM2 Java reference runtime.
 *
 * <p>The reference runtime remains correctness-first, but dense matrix-vector operations dominate the observed
 * runtime profile. This helper keeps the scalar fallback in one place and uses the Java Vector API when the
 * Workbench was launched with {@code --add-modules=jdk.incubator.vector}.</p>
 */
final class SmolLM2ReferenceDenseOps {

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
        PARALLEL_DENSE = Boolean.parseBoolean(
                System.getProperty("windirectml.smollm2.reference.parallelDense", "true"));
        PARALLEL_ROW_THRESHOLD = Integer.getInteger(
                "windirectml.smollm2.reference.parallelRowsThreshold", 1024);
        PARALLEL_COLUMN_THRESHOLD = Integer.getInteger(
                "windirectml.smollm2.reference.parallelColumnsThreshold", 256);
    }

    private SmolLM2ReferenceDenseOps() {
    }

    static boolean enabled() {
        return ENABLED;
    }

    static boolean parallelEnabled() {
        return PARALLEL_DENSE;
    }

    static float dot(float[] left, int leftOffset, float[] right, int rightOffset, int length) {
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

    static void multiplyRows(float[] matrix, int rows, int cols, float[] input, float[] output) {
        if (input.length != cols) {
            throw new IllegalArgumentException("Input width mismatch: expected " + cols + " but got " + input.length);
        }
        if (output.length != rows) {
            throw new IllegalArgumentException("Output height mismatch: expected " + rows + " but got " + output.length);
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

    static void gatedSiluMultiply(float[] gate, float[] up) {
        if (gate.length != up.length) {
            throw new IllegalArgumentException("gate and up vectors must have the same length");
        }
        for (int i = 0; i < gate.length; i++) {
            gate[i] = fastSilu(gate[i]) * up[i];
        }
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

    private static float fastSilu(float value) {
        return value / (1.0f + (float) Math.exp(-value));
    }
}
