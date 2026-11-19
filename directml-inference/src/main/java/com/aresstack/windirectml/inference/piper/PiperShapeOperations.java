package com.aresstack.windirectml.inference.piper;

import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.io.IOException;
import java.util.Arrays;

/** Implement the dynamic VITS shape constructors without model-specific assumptions. */
public final class PiperShapeOperations {

    private PiperShapeOperations() {
    }

    public static PiperTensor constantOfShape(PiperTensor requestedShape, OnnxTensor value)
            throws IOException {
        long[] dimensions = requestedShape.requireInteger();
        int[] shape = new int[dimensions.length];
        for (int axis = 0; axis < dimensions.length; axis++) {
            if (dimensions[axis] < 0 || dimensions[axis] > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("ConstantOfShape requires nonnegative dimensions");
            }
            shape[axis] = (int) dimensions[axis];
        }
        int count = PiperTensor.elementCount(shape);
        if (value == null) {
            return PiperTensor.floats(new float[count], shape);
        }
        PiperTensor scalar = PiperTensor.fromOnnx(value);
        if (scalar.length() != 1) {
            throw new IllegalArgumentException("ConstantOfShape value must be a scalar");
        }
        if (scalar.dataType() == PiperTensor.FLOAT32) {
            float[] output = new float[count];
            Arrays.fill(output, scalar.requireFloat()[0]);
            return PiperTensor.floats(output, shape);
        }
        if (scalar.dataType() == PiperTensor.INT32 || scalar.dataType() == PiperTensor.INT64) {
            long[] output = new long[count];
            Arrays.fill(output, scalar.requireInteger()[0]);
            return PiperTensor.integers(output, shape);
        }
        if (scalar.dataType() == PiperTensor.BOOL) {
            boolean[] output = new boolean[count];
            Arrays.fill(output, scalar.requireBoolean()[0]);
            return PiperTensor.bools(output, shape);
        }
        throw new IllegalArgumentException("Unsupported ConstantOfShape value type");
    }

    public static PiperTensor expand(PiperTensor input, PiperTensor requested) {
        long[] dimensions = requested.requireInteger();
        int[] original = input.shape();
        int rank = Math.max(original.length, dimensions.length);
        int[] target = new int[rank];
        int[] source = new int[rank];
        Arrays.fill(target, 1);
        Arrays.fill(source, 1);
        System.arraycopy(original, 0, source, rank - original.length, original.length);
        for (int i = 0; i < dimensions.length; i++) {
            long size = dimensions[i];
            if (size < 0 || size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Expand requires nonnegative dimensions");
            }
            target[rank - dimensions.length + i] = (int) size;
        }
        for (int axis = 0; axis < rank; axis++) {
            if (source[axis] != target[axis] && source[axis] != 1 && target[axis] != 1) {
                throw new IllegalArgumentException("Expand dimensions are incompatible");
            }
            if (source[axis] == 1) {
                target[axis] = target[axis] == 0 ? 0 : Math.max(1, target[axis]);
            } else {
                target[axis] = source[axis];
            }
        }
        int count = PiperTensor.elementCount(target);
        int[] outputStride = strides(target);
        int[] sourceStride = strides(source);
        for (int axis = 0; axis < rank; axis++) {
            if (source[axis] == 1) {
                sourceStride[axis] = 0;
            }
        }

        if (input.dataType() == PiperTensor.FLOAT32) {
            float[] data = input.requireFloat();
            float[] result = new float[count];
            for (int i = 0; i < count; i++) {
                result[i] = data[broadcastOffset(i, outputStride, sourceStride)];
            }
            return PiperTensor.floats(result, target);
        }
        if (input.dataType() == PiperTensor.BOOL) {
            boolean[] data = input.requireBoolean();
            boolean[] result = new boolean[count];
            for (int i = 0; i < count; i++) {
                result[i] = data[broadcastOffset(i, outputStride, sourceStride)];
            }
            return PiperTensor.bools(result, target);
        }
        long[] data = input.requireInteger();
        long[] result = new long[count];
        for (int i = 0; i < count; i++) {
            result[i] = data[broadcastOffset(i, outputStride, sourceStride)];
        }
        return PiperTensor.integers(result, target);
    }

    public static PiperTensor squeeze(PiperTensor input, PiperTensor requestedAxes) {
        int[] source = input.shape();
        boolean[] remove = new boolean[source.length];
        if (requestedAxes == null) {
            for (int axis = 0; axis < source.length; axis++) {
                remove[axis] = source[axis] == 1;
            }
        } else {
            for (long requested : requestedAxes.requireInteger()) {
                long normalized = requested < 0 ? requested + source.length : requested;
                if (normalized < 0 || normalized >= source.length || remove[(int) normalized]) {
                    throw new IllegalArgumentException("Squeeze axis is invalid or repeated");
                }
                if (source[(int) normalized] != 1) {
                    throw new IllegalArgumentException("Squeeze requires a singleton dimension");
                }
                remove[(int) normalized] = true;
            }
        }
        int remaining = 0;
        for (boolean removed : remove) {
            if (!removed) {
                remaining++;
            }
        }
        int[] shape = new int[remaining];
        int next = 0;
        for (int i = 0; i < source.length; i++) {
            if (!remove[i]) {
                shape[next++] = source[i];
            }
        }
        return PiperTensorOperations.reshape(input, shape);
    }

    private static int[] strides(int[] dimensions) {
        int[] result = new int[dimensions.length];
        int current = 1;
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            result[axis] = current;
            current = Math.multiplyExact(current, dimensions[axis]);
        }
        return result;
    }

    private static int broadcastOffset(int flatIndex, int[] outputStride, int[] sourceStride) {
        int remaining = flatIndex;
        int result = 0;
        for (int axis = 0; axis < outputStride.length; axis++) {
            int index = remaining / outputStride[axis];
            remaining %= outputStride[axis];
            result += index * sourceStride[axis];
        }
        return result;
    }
}
