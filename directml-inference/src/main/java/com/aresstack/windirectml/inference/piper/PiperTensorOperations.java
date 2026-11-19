package com.aresstack.windirectml.inference.piper;

import java.util.Arrays;
import java.util.List;

/**
 * Execute shape, indexing and arithmetic operators of the VITS reference graph.
 * Keep these kernels deterministic so DirectML implementations can use them as parity oracles.
 */
public final class PiperTensorOperations {

    private PiperTensorOperations() {
    }

    public static List<PiperTensor> split(PiperTensor source, int requestedAxis,
                                          long[] requestedSizes, int outputCount) {
        int axis = normalizeAxis(requestedAxis, source.rank());
        int total = source.shape()[axis];
        if (outputCount < 1) {
            throw new IllegalArgumentException("Split requires at least one output");
        }
        long[] sizes = requestedSizes == null ? new long[outputCount] : requestedSizes.clone();
        if (requestedSizes == null) {
            int base = total / outputCount;
            int remainder = total % outputCount;
            for (int i = 0; i < outputCount; i++) {
                sizes[i] = base + (i < remainder ? 1 : 0);
            }
        }
        if (sizes.length != outputCount) {
            throw new IllegalArgumentException("Split outputs do not match the specified sizes");
        }
        long sum = 0;
        for (long size : sizes) {
            if (size < 0 || size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Invalid Split part length");
            }
            sum += size;
        }
        if (sum != total) {
            throw new IllegalArgumentException("Split part lengths must cover the axis");
        }
        List<PiperTensor> result = new java.util.ArrayList<>(outputCount);
        long position = 0;
        PiperTensor axes = PiperTensor.integers(new long[]{axis}, 1);
        PiperTensor steps = PiperTensor.integers(new long[]{1}, 1);
        for (long size : sizes) {
            result.add(slice(source, PiperTensor.integers(new long[]{position}),
                    PiperTensor.integers(new long[]{position + size}), axes, steps));
            position += size;
        }
        return java.util.List.copyOf(result);
    }

    public static PiperTensor reduceMean(PiperTensor input, List<Long> requestedAxes,
                                         boolean keepDimensions) {
        float[] values = input.requireFloat();
        int[] inputShape = input.shape();
        boolean[] reduced = new boolean[inputShape.length];
        if (requestedAxes.isEmpty()) {
            Arrays.fill(reduced, true);
        } else {
            for (long requestedAxis : requestedAxes) {
                int axis = normalizeAxis(Math.toIntExact(requestedAxis), inputShape.length);
                if (reduced[axis]) {
                    throw new IllegalArgumentException("Repeated ReduceMean axis");
                }
                reduced[axis] = true;
            }
        }
        int remainingAxes = 0;
        long reductionSize = 1;
        for (int axis = 0; axis < inputShape.length; axis++) {
            if (reduced[axis]) {
                reductionSize *= inputShape[axis];
            } else {
                remainingAxes++;
            }
        }
        int[] outputShape = new int[keepDimensions ? inputShape.length : remainingAxes];
        int write = 0;
        for (int axis = 0; axis < inputShape.length; axis++) {
            if (keepDimensions) {
                outputShape[write++] = reduced[axis] ? 1 : inputShape[axis];
            } else if (!reduced[axis]) {
                outputShape[write++] = inputShape[axis];
            }
        }
        int outputCount = PiperTensor.elementCount(outputShape);
        int[] inputStrides = strides(inputShape);
        int[] outputStrides = strides(outputShape);
        float[] output = new float[outputCount];
        for (int index = 0; index < values.length; index++) {
            int offset = 0;
            int remaining = index;
            int outputAxis = 0;
            for (int axis = 0; axis < inputShape.length; axis++) {
                int coordinate = remaining / inputStrides[axis];
                remaining %= inputStrides[axis];
                if (keepDimensions || !reduced[axis]) {
                    offset += (reduced[axis] ? 0 : coordinate) * outputStrides[outputAxis];
                    outputAxis++;
                }
            }
            output[offset] += values[index];
        }
        if (reductionSize < 1) {
            throw new IllegalArgumentException("ReduceMean has empty reduction dimensions");
        }
        for (int index = 0; index < output.length; index++) {
            output[index] /= reductionSize;
        }
        return PiperTensor.floats(output, outputShape);
    }

    public static PiperTensor unary(String operator, PiperTensor input) {
        float[] values = input.requireFloat();
        float[] result = new float[values.length];
        for (int index = 0; index < values.length; index++) {
            float value = values[index];
            result[index] = switch (operator) {
                case "Sqrt" -> (float) Math.sqrt(value);
                case "Erf" -> approximateErf(value);
                case "Relu" -> Math.max(0f, value);
                case "Exp" -> (float) Math.exp(value);
                case "Tanh" -> (float) Math.tanh(value);
                case "Sigmoid" -> (float) (1d / (1d + Math.exp(-value)));
                case "Neg" -> -value;
                default -> throw new IllegalArgumentException("Unsupported Piper unary operation " + operator);
            };
        }
        return PiperTensor.floats(result, input.shape());
    }

    public static PiperTensor equal(PiperTensor first, PiperTensor second) {
        int[] outputShape = broadcastShape(first.shape(), second.shape());
        int[] outputStride = strides(outputShape);
        int[] aStride = broadcastStrides(first.shape(), outputShape);
        int[] bStride = broadcastStrides(second.shape(), outputShape);
        int count = PiperTensor.elementCount(outputShape);
        boolean[] result = new boolean[count];
        for (int index = 0; index < count; index++) {
            int a = broadcastOffset(index, outputStride, aStride);
            int b = broadcastOffset(index, outputStride, bStride);
            if (first.dataType() == PiperTensor.BOOL && second.dataType() == PiperTensor.BOOL) {
                result[index] = first.requireBoolean()[a] == second.requireBoolean()[b];
            } else {
                result[index] = numericValue(first, a) == numericValue(second, b);
            }
        }
        return PiperTensor.bools(result, outputShape);
    }

    public static PiperTensor where(PiperTensor condition, PiperTensor positive, PiperTensor negative) {
        if (condition.dataType() != PiperTensor.BOOL
                || positive.dataType() != negative.dataType()) {
            throw new IllegalArgumentException("Piper Where requires BOOL condition and matching data types");
        }
        int[] outputShape = broadcastShape(broadcastShape(condition.shape(),
                positive.shape()), negative.shape());
        int[] outputStride = strides(outputShape);
        int[] conditionStrides = broadcastStrides(condition.shape(), outputShape);
        int[] positiveStrides = broadcastStrides(positive.shape(), outputShape);
        int[] negativeStrides = broadcastStrides(negative.shape(), outputShape);
        int count = PiperTensor.elementCount(outputShape);
        boolean[] flags = condition.requireBoolean();
        if (positive.dataType() == PiperTensor.FLOAT32) {
            float[] x = positive.requireFloat();
            float[] y = negative.requireFloat();
            float[] output = new float[count];
            for (int index = 0; index < count; index++) {
                output[index] = flags[broadcastOffset(index, outputStride, conditionStrides)]
                        ? x[broadcastOffset(index, outputStride, positiveStrides)]
                        : y[broadcastOffset(index, outputStride, negativeStrides)];
            }
            return PiperTensor.floats(output, outputShape);
        }
        if (positive.dataType() == PiperTensor.BOOL) {
            boolean[] x = positive.requireBoolean();
            boolean[] y = negative.requireBoolean();
            boolean[] output = new boolean[count];
            for (int index = 0; index < count; index++) {
                output[index] = flags[broadcastOffset(index, outputStride, conditionStrides)]
                        ? x[broadcastOffset(index, outputStride, positiveStrides)]
                        : y[broadcastOffset(index, outputStride, negativeStrides)];
            }
            return PiperTensor.bools(output, outputShape);
        }
        long[] x = positive.requireInteger();
        long[] y = negative.requireInteger();
        long[] output = new long[count];
        for (int index = 0; index < count; index++) {
            output[index] = flags[broadcastOffset(index, outputStride, conditionStrides)]
                    ? x[broadcastOffset(index, outputStride, positiveStrides)]
                    : y[broadcastOffset(index, outputStride, negativeStrides)];
        }
        return PiperTensor.integers(output, outputShape);
    }

    public static PiperTensor softmax(PiperTensor input, int requestedAxis) {
        int axis = normalizeAxis(requestedAxis, input.rank());
        int[] shape = input.shape();
        int outer = product(shape, 0, axis);
        int channels = shape[axis];
        int inner = product(shape, axis + 1, shape.length);
        float[] values = input.requireFloat();
        float[] result = new float[values.length];
        for (int row = 0; row < outer; row++) {
            for (int column = 0; column < inner; column++) {
                int first = row * channels * inner + column;
                float maximum = Float.NEGATIVE_INFINITY;
                for (int position = 0; position < channels; position++) {
                    maximum = Math.max(maximum, values[first + position * inner]);
                }
                double sum = 0;
                for (int position = 0; position < channels; position++) {
                    int offset = first + position * inner;
                    result[offset] = (float) Math.exp(values[offset] - maximum);
                    sum += result[offset];
                }
                for (int position = 0; position < channels; position++) {
                    result[first + position * inner] /= (float) sum;
                }
            }
        }
        return PiperTensor.floats(result, shape);
    }

    public static PiperTensor slice(PiperTensor source, PiperTensor starts, PiperTensor ends,
                                    PiperTensor requestedAxes, PiperTensor requestedSteps) {
        long[] from = starts.requireInteger();
        long[] to = ends.requireInteger();
        long[] axes = requestedAxes.requireInteger();
        long[] steps = requestedSteps.requireInteger();
        if (from.length != to.length || from.length != axes.length || from.length != steps.length) {
            throw new IllegalArgumentException("Slice coordinates must have the same count");
        }
        int[] shape = source.shape();
        int[] initial = new int[shape.length];
        int[] stride = new int[shape.length];
        Arrays.fill(stride, 1);
        int[] resultShape = shape.clone();
        boolean[] seen = new boolean[shape.length];
        for (int i = 0; i < from.length; i++) {
            int axis = normalizeAxis(Math.toIntExact(axes[i]), shape.length);
            if (seen[axis] || steps[i] == 0 || steps[i] > Integer.MAX_VALUE
                    || steps[i] < Integer.MIN_VALUE) {
                throw new IllegalArgumentException("Invalid or duplicated Slice axis");
            }
            seen[axis] = true;
            stride[axis] = (int) steps[i];
            long dimension = shape[axis];
            if (stride[axis] > 0) {
                long start = from[i] < 0 ? from[i] + dimension : from[i];
                long end = to[i] < 0 ? to[i] + dimension : to[i];
                initial[axis] = (int) Math.min(Math.max(start, 0), dimension);
                long normalizedEnd = Math.min(Math.max(end, 0), dimension);
                resultShape[axis] = (int) Math.max(0,
                        (normalizedEnd - initial[axis] + stride[axis] - 1) / stride[axis]);
            } else {
                long start = from[i] < 0 ? from[i] + dimension : from[i];
                long end = to[i] < 0 ? to[i] + dimension : to[i];
                initial[axis] = (int) Math.min(Math.max(start, -1), dimension - 1);
                long normalizedEnd = Math.min(Math.max(end, -1), dimension - 1);
                long count = initial[axis] <= normalizedEnd ? 0
                        : (initial[axis] - normalizedEnd - stride[axis] - 1) / -stride[axis];
                resultShape[axis] = Math.toIntExact(count);
            }
        }
        int length = PiperTensor.elementCount(resultShape);
        int[] sourceStride = strides(shape);
        int[] outputStride = strides(resultShape);
        if (source.dataType() == PiperTensor.FLOAT32) {
            float[] data = source.requireFloat();
            float[] output = new float[length];
            for (int index = 0; index < length; index++) {
                output[index] = data[transformedOffset(index, outputStride, sourceStride,
                        initial, stride)];
            }
            return PiperTensor.floats(output, resultShape);
        }
        if (source.dataType() == PiperTensor.BOOL) {
            boolean[] data = source.requireBoolean();
            boolean[] output = new boolean[length];
            for (int index = 0; index < length; index++) {
                output[index] = data[transformedOffset(index, outputStride, sourceStride,
                        initial, stride)];
            }
            return PiperTensor.bools(output, resultShape);
        }
        long[] data = source.requireInteger();
        long[] output = new long[length];
        for (int index = 0; index < length; index++) {
            output[index] = data[transformedOffset(index, outputStride, sourceStride,
                    initial, stride)];
        }
        return PiperTensor.integers(output, resultShape);
    }

    public static PiperTensor pad(PiperTensor input, PiperTensor requestedPads) {
        long[] pads = requestedPads.requireInteger();
        int[] sourceShape = input.shape();
        if (pads.length != sourceShape.length * 2) {
            throw new IllegalArgumentException("Pad requires start and end coordinates for every axis");
        }
        int[] starts = new int[sourceShape.length];
        int[] outputShape = new int[sourceShape.length];
        for (int axis = 0; axis < sourceShape.length; axis++) {
            starts[axis] = Math.toIntExact(pads[axis]);
            outputShape[axis] = Math.toIntExact((long) sourceShape[axis]
                    + pads[axis] + pads[axis + sourceShape.length]);
            if (outputShape[axis] < 0) {
                throw new IllegalArgumentException("Padding crops beyond tensor boundaries");
            }
        }
        int count = PiperTensor.elementCount(outputShape);
        int[] sourceStrides = strides(sourceShape);
        int[] outputStrides = strides(outputShape);
        if (input.dataType() == PiperTensor.FLOAT32) {
            float[] data = input.requireFloat();
            float[] output = new float[count];
            for (int i = 0; i < count; i++) {
                int offset = paddedOffset(i, outputStrides, sourceStrides, sourceShape, starts);
                if (offset >= 0) {
                    output[i] = data[offset];
                }
            }
            return PiperTensor.floats(output, outputShape);
        }
        if (input.dataType() == PiperTensor.BOOL) {
            boolean[] data = input.requireBoolean();
            boolean[] output = new boolean[count];
            for (int i = 0; i < count; i++) {
                int offset = paddedOffset(i, outputStrides, sourceStrides, sourceShape, starts);
                if (offset >= 0) {
                    output[i] = data[offset];
                }
            }
            return PiperTensor.bools(output, outputShape);
        }
        long[] data = input.requireInteger();
        long[] output = new long[count];
        for (int i = 0; i < count; i++) {
            int offset = paddedOffset(i, outputStrides, sourceStrides, sourceShape, starts);
            if (offset >= 0) {
                output[i] = data[offset];
            }
        }
        return PiperTensor.integers(output, outputShape);
    }

    private static int transformedOffset(int outputIndex, int[] outputStrides, int[] sourceStrides,
                                         int[] starts, int[] steps) {
        int remaining = outputIndex;
        int offset = 0;
        for (int axis = 0; axis < sourceStrides.length; axis++) {
            int coordinate = remaining / outputStrides[axis];
            remaining %= outputStrides[axis];
            offset += (starts[axis] + coordinate * steps[axis]) * sourceStrides[axis];
        }
        return offset;
    }

    private static int paddedOffset(int outputIndex, int[] outputStrides, int[] sourceStrides,
                                    int[] sourceShape, int[] starts) {
        int remaining = outputIndex;
        int offset = 0;
        for (int axis = 0; axis < sourceStrides.length; axis++) {
            int coordinate = remaining / outputStrides[axis] - starts[axis];
            remaining %= outputStrides[axis];
            if (coordinate < 0 || coordinate >= sourceShape[axis]) {
                return -1;
            }
            offset += coordinate * sourceStrides[axis];
        }
        return offset;
    }

    public static PiperTensor concat(List<PiperTensor> tensors, int requestedAxis) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("Concat requires input tensors");
        }
        int rank = tensors.get(0).rank();
        int type = tensors.get(0).dataType();
        int axis = normalizeAxis(requestedAxis, rank);
        int[] outputShape = tensors.get(0).shape();
        outputShape[axis] = 0;
        for (PiperTensor tensor : tensors) {
            int[] dimensions = tensor.shape();
            if (tensor.rank() != rank || tensor.dataType() != type) {
                throw new IllegalArgumentException("Concat tensors must have equal rank and data type");
            }
            for (int dimension = 0; dimension < rank; dimension++) {
                if (dimension != axis && dimensions[dimension] != outputShape[dimension]) {
                    throw new IllegalArgumentException("Concat dimensions must match outside the axis");
                }
            }
            outputShape[axis] = Math.addExact(outputShape[axis], dimensions[axis]);
        }
        int outer = product(outputShape, 0, axis);
        int inner = product(outputShape, axis + 1, rank);
        int total = PiperTensor.elementCount(outputShape);
        if (type == PiperTensor.FLOAT32) {
            float[] joined = new float[total];
            for (int row = 0; row < outer; row++) {
                int cursor = row * outputShape[axis] * inner;
                for (PiperTensor tensor : tensors) {
                    int block = tensor.shape()[axis] * inner;
                    System.arraycopy(tensor.requireFloat(), row * block, joined, cursor, block);
                    cursor += block;
                }
            }
            return PiperTensor.floats(joined, outputShape);
        }
        if (type == PiperTensor.BOOL) {
            boolean[] joined = new boolean[total];
            for (int row = 0; row < outer; row++) {
                int cursor = row * outputShape[axis] * inner;
                for (PiperTensor tensor : tensors) {
                    int block = tensor.shape()[axis] * inner;
                    System.arraycopy(tensor.requireBoolean(), row * block, joined, cursor, block);
                    cursor += block;
                }
            }
            return PiperTensor.bools(joined, outputShape);
        }
        long[] joined = new long[total];
        for (int row = 0; row < outer; row++) {
            int cursor = row * outputShape[axis] * inner;
            for (PiperTensor tensor : tensors) {
                int block = tensor.shape()[axis] * inner;
                System.arraycopy(tensor.requireInteger(), row * block, joined, cursor, block);
                cursor += block;
            }
        }
        return PiperTensor.integers(joined, outputShape);
    }

    public static PiperTensor reshapeOnnx(PiperTensor input, PiperTensor requested, boolean allowZero) {
        long[] target = requested.requireInteger();
        int[] source = input.shape();
        int[] output = new int[target.length];
        int inferred = -1;
        long known = 1;
        for (int axis = 0; axis < target.length; axis++) {
            long dimension = target[axis];
            if (dimension == 0 && !allowZero) {
                if (axis >= source.length) {
                    throw new IllegalArgumentException("Reshape zero axis exceeds source rank");
                }
                dimension = source[axis];
            }
            if (dimension == -1) {
                if (inferred >= 0) {
                    throw new IllegalArgumentException("Reshape can infer only one dimension");
                }
                inferred = axis;
                output[axis] = -1;
                continue;
            }
            if (dimension < 0 || dimension > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Invalid Reshape dimension");
            }
            output[axis] = (int) dimension;
            known = Math.multiplyExact(known, dimension);
        }
        if (inferred >= 0) {
            if (known == 0 || input.length() % known != 0
                    || input.length() / known > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Reshape inferred dimension cannot be resolved");
            }
            output[inferred] = (int) (input.length() / known);
        }
        return reshape(input, output);
    }

    public static PiperTensor matmul(PiperTensor left, PiperTensor right) {
        if (left.dataType() != PiperTensor.FLOAT32 || right.dataType() != PiperTensor.FLOAT32
                || left.rank() < 2 || right.rank() < 2) {
            throw new IllegalArgumentException("Piper reference MatMul requires rank >= 2 float tensors");
        }
        int[] aShape = left.shape();
        int[] bShape = right.shape();
        int rows = aShape[aShape.length - 2];
        int shared = aShape[aShape.length - 1];
        int columns = bShape[bShape.length - 1];
        if (bShape[bShape.length - 2] != shared) {
            throw new IllegalArgumentException("MatMul contracting dimensions differ");
        }
        int[] aBatch = Arrays.copyOf(aShape, aShape.length - 2);
        int[] bBatch = Arrays.copyOf(bShape, bShape.length - 2);
        int[] batchShape = broadcastShape(aBatch, bBatch);
        int[] outputShape = Arrays.copyOf(batchShape, batchShape.length + 2);
        outputShape[batchShape.length] = rows;
        outputShape[batchShape.length + 1] = columns;
        int[] batchStrides = strides(batchShape);
        int[] aStrides = broadcastStrides(aBatch, batchShape);
        int[] bStrides = broadcastStrides(bBatch, batchShape);
        int batches = PiperTensor.elementCount(batchShape);
        float[] output = new float[PiperTensor.elementCount(outputShape)];
        float[] a = left.requireFloat();
        float[] bb = right.requireFloat();
        for (int batch = 0; batch < batches; batch++) {
            int aBase = broadcastOffset(batch, batchStrides, aStrides) * rows * shared;
            int bBase = broadcastOffset(batch, batchStrides, bStrides) * shared * columns;
            int destination = batch * rows * columns;
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    float sum = 0;
                    for (int dimension = 0; dimension < shared; dimension++) {
                        sum += a[aBase + row * shared + dimension]
                                * bb[bBase + dimension * columns + column];
                    }
                    output[destination + row * columns + column] = sum;
                }
            }
        }
        return PiperTensor.floats(output, outputShape);
    }

    public static PiperTensor shape(PiperTensor input) {
        int[] source = input.shape();
        long[] dimensions = new long[source.length];
        for (int i = 0; i < source.length; i++) {
            dimensions[i] = source[i];
        }
        return PiperTensor.integers(dimensions, dimensions.length);
    }

    public static PiperTensor gather(PiperTensor source, PiperTensor positions, int requestedAxis) {
        int[] sourceShape = source.shape();
        int axis = normalizeAxis(requestedAxis, source.rank());
        int[] indexShape = positions.shape();
        long[] indices = positions.requireInteger();
        int[] outputShape = new int[source.rank() - 1 + indexShape.length];
        System.arraycopy(sourceShape, 0, outputShape, 0, axis);
        System.arraycopy(indexShape, 0, outputShape, axis, indexShape.length);
        System.arraycopy(sourceShape, axis + 1, outputShape, axis + indexShape.length,
                source.rank() - axis - 1);
        int outer = product(sourceShape, 0, axis);
        int inner = product(sourceShape, axis + 1, sourceShape.length);
        int sourceAxisSize = sourceShape[axis];
        int blockCount = indices.length;
        int outputLength = PiperTensor.elementCount(outputShape);

        if (source.dataType() == PiperTensor.FLOAT32) {
            float[] input = source.requireFloat();
            float[] output = new float[outputLength];
            copyGatherIndices(indices, sourceAxisSize, outer, inner, blockCount,
                    (from, to, length) -> System.arraycopy(input, from, output, to, length));
            return PiperTensor.floats(output, outputShape);
        }
        if (source.dataType() == PiperTensor.BOOL) {
            boolean[] input = source.requireBoolean();
            boolean[] output = new boolean[outputLength];
            copyGatherIndices(indices, sourceAxisSize, outer, inner, blockCount,
                    (from, to, length) -> System.arraycopy(input, from, output, to, length));
            return PiperTensor.bools(output, outputShape);
        }
        long[] input = source.requireInteger();
        long[] output = new long[outputLength];
        copyGatherIndices(indices, sourceAxisSize, outer, inner, blockCount,
                (from, to, length) -> System.arraycopy(input, from, output, to, length));
        return PiperTensor.integers(output, outputShape);
    }

    public static PiperTensor transpose(PiperTensor input, List<Long> requestedPerm) {
        int rank = input.rank();
        int[] permutation = new int[rank];
        if (requestedPerm.isEmpty()) {
            for (int i = 0; i < rank; i++) {
                permutation[i] = rank - i - 1;
            }
        } else {
            if (requestedPerm.size() != rank) {
                throw new IllegalArgumentException("Transpose permutation rank mismatch");
            }
            boolean[] used = new boolean[rank];
            for (int i = 0; i < rank; i++) {
                int axis = Math.toIntExact(requestedPerm.get(i));
                if (axis < 0 || axis >= rank || used[axis]) {
                    throw new IllegalArgumentException("Transpose permutation contains invalid axis");
                }
                used[axis] = true;
                permutation[i] = axis;
            }
        }

        int[] sourceShape = input.shape();
        int[] outputShape = new int[rank];
        for (int i = 0; i < rank; i++) {
            outputShape[i] = sourceShape[permutation[i]];
        }
        int[] sourceStride = strides(sourceShape);
        int[] outputStride = strides(outputShape);
        int length = input.length();
        if (input.dataType() == PiperTensor.FLOAT32) {
            float[] values = input.requireFloat();
            float[] output = new float[length];
            for (int index = 0; index < length; index++) {
                output[index] = values[permutedIndex(index, permutation, outputStride, sourceStride)];
            }
            return PiperTensor.floats(output, outputShape);
        }
        if (input.dataType() == PiperTensor.BOOL) {
            boolean[] values = input.requireBoolean();
            boolean[] output = new boolean[length];
            for (int index = 0; index < length; index++) {
                output[index] = values[permutedIndex(index, permutation, outputStride, sourceStride)];
            }
            return PiperTensor.bools(output, outputShape);
        }
        long[] values = input.requireInteger();
        long[] output = new long[length];
        for (int index = 0; index < length; index++) {
            output[index] = values[permutedIndex(index, permutation, outputStride, sourceStride)];
        }
        return PiperTensor.integers(output, outputShape);
    }

    public static PiperTensor unsqueeze(PiperTensor input, PiperTensor axesTensor) {
        long[] requestedAxes = axesTensor.requireInteger();
        int newRank = input.rank() + requestedAxes.length;
        if (newRank > 32) {
            throw new IllegalArgumentException("VITS tensor rank exceeds supported limit");
        }
        boolean[] inserted = new boolean[newRank];
        for (long requested : requestedAxes) {
            int axis = normalizeAxis(Math.toIntExact(requested), newRank);
            if (inserted[axis]) {
                throw new IllegalArgumentException("Unsqueeze axis listed twice");
            }
            inserted[axis] = true;
        }
        int[] outputShape = new int[newRank];
        int index = 0;
        int[] inputShape = input.shape();
        for (int axis = 0; axis < newRank; axis++) {
            outputShape[axis] = inserted[axis] ? 1 : inputShape[index++];
        }
        return reshape(input, outputShape);
    }

    public static PiperTensor reshape(PiperTensor input, int... requestedShape) {
        int[] outputShape = requestedShape.clone();
        if (input.length() != PiperTensor.elementCount(outputShape)) {
            throw new IllegalArgumentException("Reshape must preserve tensor element count");
        }
        if (input.dataType() == PiperTensor.FLOAT32) {
            return PiperTensor.floats(input.requireFloat(), outputShape);
        }
        if (input.dataType() == PiperTensor.BOOL) {
            return PiperTensor.bools(input.requireBoolean(), outputShape);
        }
        return PiperTensor.integers(input.requireInteger(), outputShape);
    }

    public static PiperTensor cast(PiperTensor input, int targetType) {
        if (targetType == PiperTensor.FLOAT32) {
            float[] output = new float[input.length()];
            for (int i = 0; i < output.length; i++) {
                output[i] = input.dataType() == PiperTensor.FLOAT32 ? input.requireFloat()[i]
                        : input.dataType() == PiperTensor.BOOL
                        ? (input.requireBoolean()[i] ? 1f : 0f) : input.requireInteger()[i];
            }
            return PiperTensor.floats(output, input.shape());
        }
        if (targetType == PiperTensor.INT64 || targetType == PiperTensor.INT32) {
            long[] output = new long[input.length()];
            for (int i = 0; i < output.length; i++) {
                output[i] = input.dataType() == PiperTensor.FLOAT32 ? (long) input.requireFloat()[i]
                        : input.dataType() == PiperTensor.BOOL
                        ? (input.requireBoolean()[i] ? 1 : 0) : input.requireInteger()[i];
            }
            return PiperTensor.integers(output, input.shape());
        }
        if (targetType == PiperTensor.BOOL) {
            boolean[] output = new boolean[input.length()];
            for (int i = 0; i < output.length; i++) {
                output[i] = input.dataType() == PiperTensor.FLOAT32 ? input.requireFloat()[i] != 0
                        : input.dataType() == PiperTensor.BOOL
                        ? input.requireBoolean()[i] : input.requireInteger()[i] != 0;
            }
            return PiperTensor.bools(output, input.shape());
        }
        throw new IllegalArgumentException("Unsupported Piper Cast target " + targetType);
    }

    public static PiperTensor range(PiperTensor start, PiperTensor limit, PiperTensor step) {
        long from = scalarInteger(start);
        long until = scalarInteger(limit);
        long delta = scalarInteger(step);
        if (delta == 0) {
            throw new IllegalArgumentException("Piper Range step must not be zero");
        }
        long count = delta > 0
                ? Math.max(0, (until - from + delta - 1) / delta)
                : Math.max(0, (from - until - delta - 1) / -delta);
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Piper Range is too large");
        }
        long[] values = new long[(int) count];
        for (int i = 0; i < values.length; i++) {
            values[i] = from + i * delta;
        }
        return PiperTensor.integers(values, values.length);
    }

    public static PiperTensor binary(String operation, PiperTensor first, PiperTensor second) {
        int[] outputShape = broadcastShape(first.shape(), second.shape());
        int[] outputStride = strides(outputShape);
        int[] firstStride = broadcastStrides(first.shape(), outputShape);
        int[] secondStride = broadcastStrides(second.shape(), outputShape);
        int count = PiperTensor.elementCount(outputShape);
        if ("Less".equals(operation)) {
            boolean[] output = new boolean[count];
            for (int i = 0; i < count; i++) {
                int x = broadcastOffset(i, outputStride, firstStride);
                int y = broadcastOffset(i, outputStride, secondStride);
                output[i] = numericValue(first, x) < numericValue(second, y);
            }
            return PiperTensor.bools(output, outputShape);
        }
        if (first.dataType() == PiperTensor.FLOAT32 && second.dataType() == PiperTensor.FLOAT32) {
            float[] left = first.requireFloat();
            float[] right = second.requireFloat();
            float[] output = new float[count];
            for (int i = 0; i < count; i++) {
                float a = left[broadcastOffset(i, outputStride, firstStride)];
                float b = right[broadcastOffset(i, outputStride, secondStride)];
                output[i] = switch (operation) {
                    case "Mul" -> a * b;
                    case "Add" -> a + b;
                    case "Sub" -> a - b;
                    case "Div" -> a / b;
                    case "Pow" -> (float) Math.pow(a, b);
                    default -> throw new IllegalArgumentException("Unsupported Piper binary operator: " + operation);
                };
            }
            return PiperTensor.floats(output, outputShape);
        }
        if ((first.dataType() == PiperTensor.INT64 || first.dataType() == PiperTensor.INT32)
                && (second.dataType() == PiperTensor.INT64 || second.dataType() == PiperTensor.INT32)) {
            long[] left = first.requireInteger();
            long[] right = second.requireInteger();
            long[] output = new long[count];
            for (int i = 0; i < count; i++) {
                long a = left[broadcastOffset(i, outputStride, firstStride)];
                long b = right[broadcastOffset(i, outputStride, secondStride)];
                output[i] = switch (operation) {
                    case "Mul" -> a * b;
                    case "Add" -> a + b;
                    case "Sub" -> a - b;
                    case "Div" -> a / b;
                    default -> throw new IllegalArgumentException("Unsupported Piper binary operator: " + operation);
                };
            }
            return PiperTensor.integers(output, outputShape);
        }
        throw new IllegalArgumentException("Piper binary operation has incompatible tensor types");
    }

    private static float approximateErf(float input) {
        if (Float.isNaN(input)) {
            return Float.NaN;
        }
        if (input == 0f) {
            return 0f;
        }
        double absolute = Math.abs((double) input);
        double position = 1d / (1d + 0.3275911d * absolute);
        double polynomial = 1.061405429d;
        polynomial = polynomial * position - 1.453152027d;
        polynomial = polynomial * position + 1.421413741d;
        polynomial = polynomial * position - 0.284496736d;
        polynomial = polynomial * position + 0.254829592d;
        polynomial *= position;
        return (float) Math.copySign(1d - polynomial * Math.exp(-absolute * absolute), input);
    }

    private static double numericValue(PiperTensor tensor, int index) {
        return tensor.dataType() == PiperTensor.FLOAT32 ? tensor.requireFloat()[index]
                : tensor.requireInteger()[index];
    }

    private static int[] broadcastShape(int[] a, int[] b) {
        int rank = Math.max(a.length, b.length);
        int[] result = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            int left = axis >= rank - a.length ? a[axis - (rank - a.length)] : 1;
            int right = axis >= rank - b.length ? b[axis - (rank - b.length)] : 1;
            if (left != right && left != 1 && right != 1) {
                throw new IllegalArgumentException("Piper broadcast dimensions do not match");
            }
            result[axis] = Math.max(left, right);
        }
        return result;
    }

    private static int[] broadcastStrides(int[] inputShape, int[] outputShape) {
        int[] original = strides(inputShape);
        int[] result = new int[outputShape.length];
        int shift = outputShape.length - inputShape.length;
        for (int axis = shift; axis < outputShape.length; axis++) {
            result[axis] = inputShape[axis - shift] == 1 ? 0 : original[axis - shift];
        }
        return result;
    }

    private static int broadcastOffset(int outputIndex, int[] outputStrides, int[] inputStrides) {
        int offset = 0;
        int remaining = outputIndex;
        for (int axis = 0; axis < outputStrides.length; axis++) {
            int coordinate = remaining / outputStrides[axis];
            remaining %= outputStrides[axis];
            offset += coordinate * inputStrides[axis];
        }
        return offset;
    }

    private static int permutedIndex(int outputIndex, int[] permutation,
                                      int[] outputStrides, int[] inputStrides) {
        int offset = 0;
        int remaining = outputIndex;
        for (int axis = 0; axis < permutation.length; axis++) {
            int coordinate = remaining / outputStrides[axis];
            remaining %= outputStrides[axis];
            offset += coordinate * inputStrides[permutation[axis]];
        }
        return offset;
    }

    private static int[] strides(int[] shape) {
        int[] stride = new int[shape.length];
        int product = 1;
        for (int axis = shape.length - 1; axis >= 0; axis--) {
            stride[axis] = product;
            product = Math.multiplyExact(product, shape[axis]);
        }
        return stride;
    }

    private static int product(int[] shape, int from, int to) {
        int result = 1;
        for (int i = from; i < to; i++) {
            result = Math.multiplyExact(result, shape[i]);
        }
        return result;
    }

    private static int normalizeAxis(int requested, int rank) {
        int axis = requested < 0 ? requested + rank : requested;
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Piper axis is out of tensor bounds: " + requested);
        }
        return axis;
    }

    private static long scalarInteger(PiperTensor tensor) {
        if (tensor.length() != 1) {
            throw new IllegalArgumentException("Piper scalar operation requires one element");
        }
        return tensor.requireInteger()[0];
    }

    private static void copyGatherIndices(long[] indices, int size, int outer, int inner,
                                          int count, BlockCopy copy) {
        for (int row = 0; row < outer; row++) {
            for (int i = 0; i < count; i++) {
                long resolved = indices[i] < 0 ? indices[i] + size : indices[i];
                if (resolved < 0 || resolved >= size) {
                    throw new IllegalArgumentException("Piper Gather index outside tensor bounds");
                }
                copy.copy((row * size + (int) resolved) * inner, (row * count + i) * inner, inner);
            }
        }
    }

    @FunctionalInterface
    private interface BlockCopy {
        void copy(int from, int to, int length);
    }
}
