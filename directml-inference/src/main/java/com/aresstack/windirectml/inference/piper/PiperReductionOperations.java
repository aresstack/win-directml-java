package com.aresstack.windirectml.inference.piper;

import java.util.Arrays;

/** Execute VITS reductions and cumulative sums without external inference runtimes. */
public final class PiperReductionOperations {

    private PiperReductionOperations() {
    }

    public static PiperTensor cumSum(PiperTensor input, PiperTensor axisTensor,
                                     boolean exclusive, boolean reverse) {
        long[] requested = axisTensor.requireInteger();
        if (axisTensor.length() != 1) {
            throw new IllegalArgumentException("CumSum axis must be a scalar");
        }
        int[] shape = input.shape();
        int axis = normalizeAxis(requested[0], shape.length);
        int outer = product(shape, 0, axis);
        int axisLength = shape[axis];
        int inner = product(shape, axis + 1, shape.length);
        if (input.dataType() == PiperTensor.FLOAT32) {
            float[] source = input.requireFloat();
            float[] output = new float[source.length];
            for (int batch = 0; batch < outer; batch++) {
                for (int column = 0; column < inner; column++) {
                    float sum = 0;
                    for (int position = 0; position < axisLength; position++) {
                        int channel = reverse ? axisLength - 1 - position : position;
                        int offset = (batch * axisLength + channel) * inner + column;
                        if (exclusive) {
                            output[offset] = sum;
                            sum += source[offset];
                        } else {
                            sum += source[offset];
                            output[offset] = sum;
                        }
                    }
                }
            }
            return PiperTensor.floats(output, shape);
        }
        long[] source = input.requireInteger();
        long[] output = new long[source.length];
        for (int batch = 0; batch < outer; batch++) {
            for (int column = 0; column < inner; column++) {
                long sum = 0;
                for (int position = 0; position < axisLength; position++) {
                    int channel = reverse ? axisLength - 1 - position : position;
                    int offset = (batch * axisLength + channel) * inner + column;
                    if (exclusive) {
                        output[offset] = sum;
                        sum += source[offset];
                    } else {
                        sum += source[offset];
                        output[offset] = sum;
                    }
                }
            }
        }
        return PiperTensor.integers(output, shape);
    }

    public static PiperTensor reduceSum(PiperTensor input, long[] requestedAxes,
                                        boolean keepDimensions, boolean noopWithEmptyAxes) {
        return reduce(input, requestedAxes, keepDimensions, noopWithEmptyAxes, false);
    }

    public static PiperTensor reduceMax(PiperTensor input, long[] requestedAxes,
                                        boolean keepDimensions, boolean noopWithEmptyAxes) {
        return reduce(input, requestedAxes, keepDimensions, noopWithEmptyAxes, true);
    }

    private static PiperTensor reduce(PiperTensor input, long[] requestedAxes,
                                      boolean keepDimensions, boolean noopWithEmptyAxes,
                                      boolean maximum) {
        int[] shape = input.shape();
        if (requestedAxes.length == 0 && noopWithEmptyAxes) {
            return PiperTensorOperations.reshape(input, shape);
        }
        boolean[] reduced = new boolean[shape.length];
        if (requestedAxes.length == 0) {
            Arrays.fill(reduced, true);
        } else {
            for (long requested : requestedAxes) {
                int axis = normalizeAxis(requested, shape.length);
                if (reduced[axis]) {
                    throw new IllegalArgumentException("Reduce axes may not repeat");
                }
                reduced[axis] = true;
            }
        }
        int dimensions = 0;
        for (boolean value : reduced) {
            if (!value) {
                dimensions++;
            }
        }
        int[] outputShape = new int[keepDimensions ? shape.length : dimensions];
        int outputAxis = 0;
        for (int axis = 0; axis < shape.length; axis++) {
            if (keepDimensions) {
                outputShape[outputAxis++] = reduced[axis] ? 1 : shape[axis];
            } else if (!reduced[axis]) {
                outputShape[outputAxis++] = shape[axis];
            }
        }
        int count = PiperTensor.elementCount(outputShape);
        int[] strides = strides(shape);
        int[] outputStride = strides(outputShape);
        if (input.dataType() == PiperTensor.FLOAT32) {
            float[] result = new float[count];
            if (maximum) {
                Arrays.fill(result, Float.NEGATIVE_INFINITY);
            }
            float[] source = input.requireFloat();
            for (int index = 0; index < source.length; index++) {
                int offset = reduceOffset(index, shape, strides, outputStride, reduced,
                        keepDimensions);
                result[offset] = maximum ? Math.max(result[offset], source[index])
                        : result[offset] + source[index];
            }
            ensureMaxNotEmpty(maximum, input, result.length);
            return PiperTensor.floats(result, outputShape);
        }
        long[] result = new long[count];
        if (maximum) {
            Arrays.fill(result, Long.MIN_VALUE);
        }
        long[] source = input.requireInteger();
        for (int index = 0; index < source.length; index++) {
            int offset = reduceOffset(index, shape, strides, outputStride, reduced,
                    keepDimensions);
            result[offset] = maximum ? Math.max(result[offset], source[index])
                    : result[offset] + source[index];
        }
        ensureMaxNotEmpty(maximum, input, result.length);
        return PiperTensor.integers(result, outputShape);
    }

    private static void ensureMaxNotEmpty(boolean maximum, PiperTensor input, int outputCount) {
        if (maximum && input.length() == 0 && outputCount > 0) {
            throw new IllegalArgumentException("ReduceMax cannot reduce an empty axis");
        }
    }

    private static int reduceOffset(int index, int[] shape, int[] strides,
                                    int[] outputStride, boolean[] reduced, boolean keepDimensions) {
        int remaining = index;
        int offset = 0;
        int next = 0;
        for (int axis = 0; axis < shape.length; axis++) {
            int coordinate = remaining / strides[axis];
            remaining %= strides[axis];
            if (keepDimensions || !reduced[axis]) {
                offset += (reduced[axis] ? 0 : coordinate) * outputStride[next++];
            }
        }
        return offset;
    }

    private static int normalizeAxis(long requested, int rank) {
        long index = requested < 0 ? requested + rank : requested;
        if (index < 0 || index >= rank) {
            throw new IllegalArgumentException("Reduction axis is outside tensor rank");
        }
        return (int) index;
    }

    private static int product(int[] dimensions, int start, int end) {
        int result = 1;
        for (int index = start; index < end; index++) {
            result = Math.multiplyExact(result, dimensions[index]);
        }
        return result;
    }

    private static int[] strides(int[] dimensions) {
        int[] strides = new int[dimensions.length];
        int size = 1;
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            strides[axis] = size;
            size = Math.multiplyExact(size, dimensions[axis]);
        }
        return strides;
    }
}
