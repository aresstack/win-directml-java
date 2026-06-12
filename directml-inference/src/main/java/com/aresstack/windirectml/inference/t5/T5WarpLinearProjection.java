package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.warp.WarpDenseProjection;
import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.Objects;

/**
 * WARP/DirectML-backed T5 linear projection.
 *
 * <p>Thin T5 adapter over the shared, model-family-neutral {@link WarpDenseProjection}. T5 keeps its own
 * {@link T5LinearProjection} contract (and the encoder/decoder math stays independent from decoder-only runtimes),
 * while the actual WARP upload + matvec/batch path is the shared building block. Behaviour is unchanged from the
 * previous T5-specific implementation: same shapes, same per-row batch fallback, same matvec results.</p>
 */
public final class T5WarpLinearProjection implements T5LinearProjection {
    private final WarpDenseProjection projection;

    private T5WarpLinearProjection(WarpDenseProjection projection) {
        this.projection = Objects.requireNonNull(projection, "projection");
    }

    public static T5WarpLinearProjection from(WindowsBindings windowsBindings, T5TensorData weight) {
        Objects.requireNonNull(weight, "weight");
        if (weight.rank() != 2) {
            throw new IllegalArgumentException("WARP projection weight must be rank 2: " + weight.name());
        }
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        int outputSize = weight.dim(0);
        int inputSize = weight.dim(1);
        return new T5WarpLinearProjection(WarpDenseProjection.fromDequantizedWeights(
                windowsBindings, weight.name(), outputSize, inputSize, weight.values()));
    }

    @Override
    public String name() {
        return projection.name();
    }

    @Override
    public int inputSize() {
        return projection.inputSize();
    }

    @Override
    public int outputSize() {
        return projection.outputSize();
    }

    @Override
    public float[] apply(float[] input) {
        return projection.project(input);
    }

    @Override
    public float[] applySequence(float[] input, int sequenceLength, int inputSize) {
        if (inputSize != inputSize()) {
            throw new IllegalArgumentException("WARP projection input size mismatch for " + name()
                    + ": input=" + inputSize + ", expected=" + inputSize());
        }
        return projection.projectSequence(input, sequenceLength);
    }

    public boolean isClosed() {
        return projection.isClosed();
    }

    @Override
    public void close() {
        projection.close();
    }
}
