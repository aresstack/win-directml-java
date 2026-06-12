package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.inference.warp.WarpDenseProjection;
import com.aresstack.windirectml.windows.MatMulNBitsKernel;
import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.Objects;

/**
 * DirectML/WARP-backed dense projection for decoder-only model families.
 *
 * <p>Thin decoder-only adapter over the shared, model-family-neutral {@link WarpDenseProjection} (slice T5-2b). The
 * {@link DecoderOnlyDenseProjection} contract is unchanged for callers (Qwen/SmolLM2), while the actual WARP upload +
 * matvec/batch path is now the same shared building block T5 uses. Behaviour is identical: same kernel calls in the
 * same order, same per-row batch fallback, same {@link #kernel()} for GPU-resident subgraph chaining.</p>
 */
public final class DecoderOnlyWarpDenseProjection implements DecoderOnlyDenseProjection {
    private final WarpDenseProjection projection;

    private DecoderOnlyWarpDenseProjection(WarpDenseProjection projection) {
        this.projection = Objects.requireNonNull(projection, "projection");
    }

    public static DecoderOnlyWarpDenseProjection fromRowMajorWeights(WindowsBindings windowsBindings,
                                                                      String name,
                                                                      int outputSize,
                                                                      int inputSize,
                                                                      float[] weights) {
        return new DecoderOnlyWarpDenseProjection(WarpDenseProjection.fromDequantizedWeights(
                windowsBindings, name, outputSize, inputSize, weights));
    }

    @Override
    public String name() {
        return projection.name();
    }

    /**
     * The underlying GPU matmul kernel. Exposed so GPU-resident subgraphs (e.g. a fused MLP block) can chain this
     * projection's dispatch into a shared pipeline without an intermediate CPU readback.
     */
    public MatMulNBitsKernel kernel() {
        return projection.kernel();
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
    public float[] project(float[] input) {
        return projection.project(input);
    }

    @Override
    public void projectInto(float[] input, float[] output) {
        projection.projectInto(input, output);
    }

    @Override
    public void projectSequenceInto(float[] input, int sequenceLength, float[] output) {
        projection.projectSequenceInto(input, sequenceLength, output);
    }

    @Override
    public void close() {
        projection.close();
    }

    public boolean isClosed() {
        return projection.isClosed();
    }
}
