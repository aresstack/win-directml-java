package com.aresstack.windirectml.encoder.safetensors;

import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.runtime.TensorShape;

import java.util.Objects;

/**
 * Metadaten eines einzelnen Tensors in einer {@code .safetensors}-Datei.
 *
 * @param name        Tensor-Name (Schlüssel im JSON-Header).
 * @param dataType    Datentyp gemäß Safetensors-Spezifikation.
 * @param shape       Form-Beschreibung.
 * @param dataOffset  Offset in Bytes ab Beginn des Daten-Bereichs.
 * @param byteLength  Länge in Bytes.
 */
public record SafetensorsEntry(
        String name,
        TensorDataType dataType,
        TensorShape shape,
        long dataOffset,
        long byteLength
) {
    public SafetensorsEntry {
        Objects.requireNonNull(name);
        Objects.requireNonNull(dataType);
        Objects.requireNonNull(shape);
    }
}

