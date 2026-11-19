package com.aresstack.windirectml.inference.piper;

import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class PiperShapeOperationsTest {

    @Test
    void createsModelSpecificFloatTensorWithConstantValue() throws Exception {
        PiperTensor size = PiperTensor.integers(new long[]{1, 2, 3}, 3);
        OnnxTensor constant = new OnnxTensor("value", new long[]{1},
                PiperTensor.FLOAT32, new float[]{0.5397424f});
        PiperTensor result = PiperShapeOperations.constantOfShape(size, constant);

        assertArrayEquals(new int[]{1, 2, 3}, result.shape());
        assertArrayEquals(new float[]{0.5397424f, 0.5397424f, 0.5397424f,
                0.5397424f, 0.5397424f, 0.5397424f}, result.floatData(), 0);
    }

    @Test
    void expandsSingletonDimensionsWithoutChangingElementOrder() {
        PiperTensor source = PiperTensor.integers(new long[]{3, 5}, 1, 2, 1);
        PiperTensor expanded = PiperShapeOperations.expand(source,
                PiperTensor.integers(new long[]{2, 1, 4}, 3));

        assertArrayEquals(new int[]{2, 2, 4}, expanded.shape());
        assertArrayEquals(new long[]{3, 3, 3, 3, 5, 5, 5, 5,
                3, 3, 3, 3, 5, 5, 5, 5}, expanded.integerData());
    }

    @Test
    void squeezesSelectedSingletonAxes() {
        PiperTensor source = PiperTensor.floats(new float[]{1, 2, 3}, 1, 3, 1);
        PiperTensor result = PiperShapeOperations.squeeze(source,
                PiperTensor.integers(new long[]{-1, 0}, 2));

        assertArrayEquals(new int[]{3}, result.shape());
        assertArrayEquals(new float[]{1, 2, 3}, result.floatData(), 0);
    }

    @Test
    void rejectsIncompatibleShapes() {
        assertThrows(IllegalArgumentException.class, () ->
                PiperShapeOperations.constantOfShape(
                        PiperTensor.integers(new long[]{-1, 3}, 2), null));
        assertThrows(IllegalArgumentException.class, () ->
                PiperShapeOperations.expand(PiperTensor.floats(new float[6], 2, 3),
                        PiperTensor.integers(new long[]{4, 3}, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                PiperShapeOperations.squeeze(PiperTensor.floats(new float[6], 2, 3),
                        PiperTensor.integers(new long[]{1}, 1)));
    }
}
