package com.aresstack.windirectml.runtime;

import java.util.Objects;

/**
 * GPU-residenter Tensor: Form + Layout + Datentyp + zugrundeliegender
 * {@link GpuBuffer}.
 * <p>
 * Lebenszyklus ist an den {@link GpuBuffer} gekoppelt – das Schließen des
 * Buffers gibt auch alle abhängigen Tensoren frei.
 */
public record DirectMlTensor(TensorShape shape, TensorLayout layout, TensorDataType dataType, GpuBuffer buffer) {

    public DirectMlTensor {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(layout, "layout");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(buffer, "buffer");
    }
}

