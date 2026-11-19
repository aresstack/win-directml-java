package com.aresstack.windirectml.inference.piper;

import java.util.Arrays;

/** Execute shape-aware ONNX indexing for the native Piper reference path. */
public final class PiperIndexOperations {

    private PiperIndexOperations() {
    }

    public static PiperTensor scatterNd(PiperTensor data, PiperTensor indices, PiperTensor updates) {
        int[] dimensions = data.shape();
        int[] indexShape = indices.shape();
        if (indexShape.length < 1 || data.dataType() != updates.dataType()) {
            throw new IllegalArgumentException("ScatterND requires indices and matching update type");
        }
        int depth = indexShape[indexShape.length - 1];
        if (depth < 1 || depth > dimensions.length) {
            throw new IllegalArgumentException("ScatterND index depth is outside data rank");
        }
        int count = product(indexShape, 0, indexShape.length - 1);
        int sliceLength = product(dimensions, depth, dimensions.length);
        int[] expectedUpdateShape = new int[indexShape.length - 1 + dimensions.length - depth];
        System.arraycopy(indexShape, 0, expectedUpdateShape, 0, indexShape.length - 1);
        System.arraycopy(dimensions, depth, expectedUpdateShape, indexShape.length - 1,
                dimensions.length - depth);
        if (!Arrays.equals(expectedUpdateShape, updates.shape())) {
            throw new IllegalArgumentException("ScatterND update shape is incompatible with indices");
        }
        int[] dataStrides = strides(dimensions);
        long[] locations = indices.requireInteger();
        Object result = copyArray(data);
        Object source = rawArray(updates);
        for (int tuple = 0; tuple < count; tuple++) {
            int offset = offsetForTuple(locations, tuple * depth, depth, dimensions, dataStrides);
            System.arraycopy(source, tuple * sliceLength, result, offset, sliceLength);
        }
        return fromArray(data.dataType(), result, dimensions);
    }

    public static PiperTensor gatherNd(PiperTensor data, PiperTensor indices) {
        int[] inputShape = data.shape();
        int[] indexShape = indices.shape();
        if (indexShape.length < 1) {
            throw new IllegalArgumentException("GatherND indices must include an index dimension");
        }
        int depth = indexShape[indexShape.length - 1];
        if (depth < 1 || depth > inputShape.length) {
            throw new IllegalArgumentException("GatherND index depth is outside data rank");
        }
        int[] outputShape = new int[indexShape.length - 1 + inputShape.length - depth];
        System.arraycopy(indexShape, 0, outputShape, 0, indexShape.length - 1);
        System.arraycopy(inputShape, depth, outputShape, indexShape.length - 1,
                inputShape.length - depth);
        int sliceLength = product(inputShape, depth, inputShape.length);
        int tupleCount = product(indexShape, 0, indexShape.length - 1);
        int[] strides = strides(inputShape);
        long[] positions = indices.requireInteger();
        Object values = rawArray(data);
        Object result = emptyArray(data.dataType(), PiperTensor.elementCount(outputShape));
        for (int tuple = 0; tuple < tupleCount; tuple++) {
            int position = offsetForTuple(positions, tuple * depth, depth, inputShape, strides);
            System.arraycopy(values, position, result, tuple * sliceLength, sliceLength);
        }
        return fromArray(data.dataType(), result, outputShape);
    }

    public static PiperTensor gatherElements(PiperTensor data, PiperTensor indices, int requestedAxis) {
        int[] dimensions = data.shape();
        int[] outputShape = indices.shape();
        if (dimensions.length != outputShape.length || dimensions.length == 0) {
            throw new IllegalArgumentException("GatherElements requires same-rank input and indices");
        }
        int axis = normalizeAxis(requestedAxis, dimensions.length);
        for (int i = 0; i < dimensions.length; i++) {
            if (i != axis && outputShape[i] > dimensions[i]) {
                throw new IllegalArgumentException("GatherElements index dimensions exceed input");
            }
        }
        int[] outputStride = strides(outputShape);
        int[] inputStride = strides(dimensions);
        long[] positions = indices.requireInteger();
        Object values = rawArray(data);
        Object result = emptyArray(data.dataType(), positions.length);
        for (int position = 0; position < positions.length; position++) {
            int remaining = position;
            int dataOffset = 0;
            for (int dimension = 0; dimension < dimensions.length; dimension++) {
                int coordinate = remaining / outputStride[dimension];
                remaining %= outputStride[dimension];
                if (dimension == axis) {
                    coordinate = normalizeIndex(positions[position], dimensions[dimension]);
                }
                dataOffset += coordinate * inputStride[dimension];
            }
            System.arraycopy(values, dataOffset, result, position, 1);
        }
        return fromArray(data.dataType(), result, outputShape);
    }

    public static PiperTensor nonZero(PiperTensor data) {
        int count = 0;
        for (int index = 0; index < data.length(); index++) {
            if (isNonZero(data, index)) {
                count++;
            }
        }
        int[] dimensions = data.shape();
        long[] result = new long[Math.multiplyExact(dimensions.length, count)];
        int[] strides = strides(dimensions);
        int written = 0;
        for (int index = 0; index < data.length(); index++) {
            if (!isNonZero(data, index)) {
                continue;
            }
            int remaining = index;
            for (int axis = 0; axis < dimensions.length; axis++) {
                int coordinate = remaining / strides[axis];
                remaining %= strides[axis];
                result[axis * count + written] = coordinate;
            }
            written++;
        }
        return PiperTensor.integers(result, dimensions.length, count);
    }

    private static boolean isNonZero(PiperTensor input, int index) {
        if (input.dataType() == PiperTensor.FLOAT32) {
            return input.requireFloat()[index] != 0;
        }
        if (input.dataType() == PiperTensor.BOOL) {
            return input.requireBoolean()[index];
        }
        return input.requireInteger()[index] != 0;
    }

    private static int offsetForTuple(long[] indices, int start, int depth, int[] shape, int[] stride) {
        int result = 0;
        for (int axis = 0; axis < depth; axis++) {
            int position = normalizeIndex(indices[start + axis], shape[axis]);
            result += position * stride[axis];
        }
        return result;
    }

    private static int normalizeAxis(int requested, int rank) {
        int resolved = requested < 0 ? requested + rank : requested;
        if (resolved < 0 || resolved >= rank) {
            throw new IllegalArgumentException("Piper indexing axis is invalid");
        }
        return resolved;
    }

    private static int normalizeIndex(long index, int length) {
        long resolved = index < 0 ? index + length : index;
        if (resolved < 0 || resolved >= length) {
            throw new IllegalArgumentException("Piper index exceeds tensor bounds: " + index);
        }
        return (int) resolved;
    }

    private static int product(int[] dimensions, int start, int end) {
        int count = 1;
        for (int axis = start; axis < end; axis++) {
            count = Math.multiplyExact(count, dimensions[axis]);
        }
        return count;
    }

    private static int[] strides(int[] dimensions) {
        int[] strides = new int[dimensions.length];
        int current = 1;
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            strides[axis] = current;
            current = Math.multiplyExact(current, dimensions[axis]);
        }
        return strides;
    }

    private static Object rawArray(PiperTensor source) {
        return switch (source.dataType()) {
            case PiperTensor.FLOAT32 -> source.requireFloat();
            case PiperTensor.BOOL -> source.requireBoolean();
            case PiperTensor.INT32, PiperTensor.INT64 -> source.requireInteger();
            default -> throw new IllegalArgumentException("Unsupported indexed tensor data type");
        };
    }

    private static Object emptyArray(int type, int size) {
        return switch (type) {
            case PiperTensor.FLOAT32 -> new float[size];
            case PiperTensor.BOOL -> new boolean[size];
            case PiperTensor.INT32, PiperTensor.INT64 -> new long[size];
            default -> throw new IllegalArgumentException("Unsupported indexed tensor data type");
        };
    }

    private static Object copyArray(PiperTensor source) {
        return switch (source.dataType()) {
            case PiperTensor.FLOAT32 -> source.floatData();
            case PiperTensor.BOOL -> source.booleanData();
            case PiperTensor.INT32, PiperTensor.INT64 -> source.integerData();
            default -> throw new IllegalArgumentException("Unsupported indexed tensor data type");
        };
    }

    private static PiperTensor fromArray(int type, Object values, int... shape) {
        return switch (type) {
            case PiperTensor.FLOAT32 -> PiperTensor.floats((float[]) values, shape);
            case PiperTensor.BOOL -> PiperTensor.bools((boolean[]) values, shape);
            case PiperTensor.INT32, PiperTensor.INT64 -> PiperTensor.integers((long[]) values, shape);
            default -> throw new IllegalArgumentException("Unsupported indexed tensor data type");
        };
    }
}
