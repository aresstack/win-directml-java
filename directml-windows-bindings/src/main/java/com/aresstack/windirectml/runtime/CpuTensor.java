package com.aresstack.windirectml.runtime;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * CPU-residenter Tensor.
 * <p>
 * Hält Daten als {@link ByteBuffer}; der konkrete Datentyp ergibt sich aus
 * {@link #dataType()}. Wird sowohl für Eingabe-/Ausgabetensoren als auch
 * für Memory-mapped Gewichte verwendet (siehe Safetensors-Reader).
 */
public record CpuTensor(TensorShape shape, TensorLayout layout, TensorDataType dataType, ByteBuffer data) {

    public CpuTensor {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(layout, "layout");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(data, "data");
    }

    /**
     * Convenience: Row-Major-Float32-Tensor aus einem Float-Array.
     */
    public static CpuTensor float32(TensorShape shape, float[] values) {
        if (values.length != shape.elementCount()) {
            throw new IllegalArgumentException(
                    "values.length=" + values.length + " != elementCount=" + shape.elementCount());
        }
        ByteBuffer buf = ByteBuffer.allocateDirect(values.length * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buf.asFloatBuffer().put(values);
        return new CpuTensor(shape, TensorLayout.rowMajor(shape), TensorDataType.FLOAT32, buf);
    }
}

