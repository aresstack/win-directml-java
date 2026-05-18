package com.aresstack.windirectml.runtime;

/**
 * Physisches Speicherlayout eines Tensors.
 * <p>
 * Stride-basiert: {@code offset(idx) = sum(idx[i] * strides[i])}. Strides
 * sind in Elementen angegeben, nicht in Bytes. Damit lassen sich sowohl
 * dichte als auch nicht-zusammenhängende Layouts (Slices, transponierte
 * Views) beschreiben.
 */
public record TensorLayout(int[] strides) {

    public TensorLayout {
        if (strides == null || strides.length == 0) {
            throw new IllegalArgumentException("strides must not be empty");
        }
        strides = strides.clone();
    }

    /**
     * Kanonisches, zusammenhängendes Row-Major-Layout für die gegebene Form.
     */
    public static TensorLayout rowMajor(TensorShape shape) {
        int rank = shape.rank();
        int[] s = new int[rank];
        int stride = 1;
        for (int i = rank - 1; i >= 0; i--) {
            s[i] = stride;
            stride *= shape.dim(i);
        }
        return new TensorLayout(s);
    }
}

