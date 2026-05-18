package com.aresstack.windirectml.runtime;

import java.util.Arrays;

/**
 * Unveränderliche Tensor-Form.
 * <p>
 * Reihenfolge der Dimensionen ist die natürliche logische Reihenfolge des
 * Modells (z. B. {@code [batch, seq, hidden]}); das physische Speicherlayout
 * wird durch {@link TensorLayout} beschrieben.
 */
public record TensorShape(int[] dims) {

    public TensorShape {
        if (dims == null || dims.length == 0) {
            throw new IllegalArgumentException("dims must not be empty");
        }
        for (int d : dims) {
            if (d <= 0) throw new IllegalArgumentException("dim must be > 0, was " + d);
        }
        dims = dims.clone();
    }

    public int rank() {
        return dims.length;
    }

    public int dim(int i) {
        return dims[i];
    }

    public long elementCount() {
        long n = 1;
        for (int d : dims) n *= d;
        return n;
    }

    public static TensorShape of(int... dims) {
        return new TensorShape(dims);
    }

    @Override
    public String toString() {
        return "TensorShape" + Arrays.toString(dims);
    }
}

