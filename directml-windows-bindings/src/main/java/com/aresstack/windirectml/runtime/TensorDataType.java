package com.aresstack.windirectml.runtime;

/**
 * Datentypen für Tensoren in der gemeinsamen Runtime-Schicht.
 * <p>
 * Bewusst kleiner als der DirectML-Datentyp-Zoo: wir unterstützen heute nur
 * Formate, die von Encoder- und Decoder-Pfaden tatsächlich gebraucht werden.
 */
public enum TensorDataType {

    FLOAT32(4),
    FLOAT16(2),
    BFLOAT16(2),
    INT32(4),
    INT8(1),
    UINT8(1),
    /**
     * Gepackte 4-Bit-Quantisierung (zwei Werte pro Byte, AWQ-Layout).
     */
    INT4_PACKED(0);

    private final int byteWidth;

    TensorDataType(int byteWidth) {
        this.byteWidth = byteWidth;
    }

    /**
     * @return Bytes pro Element, oder 0 falls die Größenberechnung
     * elementweise nicht definiert ist (z. B. {@link #INT4_PACKED}).
     */
    public int byteWidth() {
        return byteWidth;
    }
}

