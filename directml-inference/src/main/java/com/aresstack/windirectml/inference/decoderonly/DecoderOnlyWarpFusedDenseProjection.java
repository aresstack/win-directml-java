package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.util.List;
import java.util.Objects;

/**
 * DirectML/WARP-backed fused dense projection shared by decoder-only model families.
 *
 * <p>Several projections that consume the <em>same</em> input vector (for example {@code q_proj}, {@code k_proj} and
 * {@code v_proj}, or {@code gate_proj} and {@code up_proj}) are stacked vertically into a single weight matrix and
 * executed as one WARP dispatch instead of several. The fused output is the row-major concatenation
 * {@code [part0 | part1 | ...]}; callers read each logical part back through {@link #copySlice} /
 * {@link #copySliceSequence}.</p>
 *
 * <p>Stacking the rows is mathematically identical to running the parts separately — every output element is computed
 * from exactly the same weights and the same input — so the fused path stays numerically aligned with the per-part
 * reference apart from GPU floating-point rounding. The win is fewer dispatch/fence round-trips, which matters most in
 * the decode path where each generated token issues many small projections.</p>
 */
public final class DecoderOnlyWarpFusedDenseProjection implements AutoCloseable {

    private final String name;
    private final int inputSize;
    private final int totalOutputSize;
    private final int[] sliceSizes;
    private final int[] sliceOffsets;
    private final DecoderOnlyWarpDenseProjection projection;

    private DecoderOnlyWarpFusedDenseProjection(String name, int inputSize, int totalOutputSize,
                                                int[] sliceSizes, int[] sliceOffsets,
                                                DecoderOnlyWarpDenseProjection projection) {
        this.name = name;
        this.inputSize = inputSize;
        this.totalOutputSize = totalOutputSize;
        this.sliceSizes = sliceSizes;
        this.sliceOffsets = sliceOffsets;
        this.projection = projection;
    }

    /**
     * Build a fused projection from row-major weight parts that all share the same input width.
     *
     * @param windowsBindings WARP device bindings
     * @param name            diagnostic name for the fused projection
     * @param inputSize       shared input width of every part
     * @param parts           ordered parts to stack vertically; each part's weights are {@code outputSize * inputSize}
     *                        floats in row-major order
     */
    public static DecoderOnlyWarpFusedDenseProjection fromRowMajorParts(WindowsBindings windowsBindings,
                                                                        String name,
                                                                        int inputSize,
                                                                        List<Part> parts) {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(parts, "parts");
        if (parts.isEmpty()) {
            throw new IllegalArgumentException("fused projection " + name + " requires at least one part");
        }
        if (inputSize < 1) {
            throw new IllegalArgumentException("inputSize must be positive: " + inputSize);
        }
        int partCount = parts.size();
        int[] sliceSizes = new int[partCount];
        int[] sliceOffsets = new int[partCount];
        int totalOutputSize = 0;
        for (int i = 0; i < partCount; i++) {
            Part part = parts.get(i);
            long expected = (long) part.outputSize * inputSize;
            if (part.weights.length != expected) {
                throw new IllegalArgumentException("fused projection " + name + " part '" + part.name
                        + "' weight length mismatch: weights=" + part.weights.length + ", expected=" + expected);
            }
            sliceSizes[i] = part.outputSize;
            sliceOffsets[i] = totalOutputSize;
            totalOutputSize = Math.addExact(totalOutputSize, part.outputSize);
        }

        // Vertical stack: row-major concatenation of the parts (part 0 rows first, then part 1, ...).
        float[] fusedWeights = new float[Math.multiplyExact(totalOutputSize, inputSize)];
        int destination = 0;
        for (Part part : parts) {
            System.arraycopy(part.weights, 0, fusedWeights, destination, part.weights.length);
            destination += part.weights.length;
        }

        DecoderOnlyWarpDenseProjection projection = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(
                windowsBindings, name, totalOutputSize, inputSize, fusedWeights);
        return new DecoderOnlyWarpFusedDenseProjection(name, inputSize, totalOutputSize,
                sliceSizes, sliceOffsets, projection);
    }

    public String name() {
        return name;
    }

    /**
     * The underlying GPU matmul kernel for the fused projection. Exposed so GPU-resident subgraphs (e.g. a fused MLP
     * block) can chain this projection's dispatch into a shared pipeline without an intermediate CPU readback.
     */
    public com.aresstack.windirectml.windows.MatMulNBitsKernel kernel() {
        return projection.kernel();
    }

    public int inputSize() {
        return inputSize;
    }

    public int totalOutputSize() {
        return totalOutputSize;
    }

    public int sliceCount() {
        return sliceSizes.length;
    }

    public int sliceSize(int sliceIndex) {
        return sliceSizes[sliceIndex];
    }

    public int sliceOffset(int sliceIndex) {
        return sliceOffsets[sliceIndex];
    }

    /** Project a single input vector, returning the fused output {@code [part0 | part1 | ...]}. */
    public float[] project(float[] input) {
        return projection.project(input);
    }

    public void projectInto(float[] input, float[] output) {
        projection.projectInto(input, output);
    }

    /**
     * Project a whole sequence in one batched dispatch. The output is packed per token:
     * {@code [token0(part0|part1|...) | token1(...) | ...]}, length {@code sequenceLength * totalOutputSize}.
     */
    public void projectSequenceInto(float[] input, int sequenceLength, float[] output) {
        projection.projectSequenceInto(input, sequenceLength, output);
    }

    /** Copy logical part {@code sliceIndex} out of a single fused output vector into {@code target}. */
    public void copySlice(float[] fusedOutput, int sliceIndex, float[] target) {
        int size = sliceSizes[sliceIndex];
        if (target.length < size) {
            throw new IllegalArgumentException("target too small for slice " + sliceIndex + " of " + name
                    + ": target=" + target.length + ", expected at least=" + size);
        }
        System.arraycopy(fusedOutput, sliceOffsets[sliceIndex], target, 0, size);
    }

    /**
     * Extract logical part {@code sliceIndex} for every token of a sequence-packed fused output into {@code target},
     * which is packed per token as {@code [token0 | token1 | ...]} with width {@link #sliceSize(int)}.
     */
    public void copySliceSequence(float[] fusedSequence, int sequenceLength, int sliceIndex, float[] target) {
        int size = sliceSizes[sliceIndex];
        int offset = sliceOffsets[sliceIndex];
        if (target.length < (long) sequenceLength * size) {
            throw new IllegalArgumentException("target too small for slice sequence " + sliceIndex + " of " + name
                    + ": target=" + target.length + ", expected at least=" + ((long) sequenceLength * size));
        }
        for (int token = 0; token < sequenceLength; token++) {
            System.arraycopy(fusedSequence, token * totalOutputSize + offset, target, token * size, size);
        }
    }

    @Override
    public void close() {
        projection.close();
    }

    /** A single weight matrix to stack into the fused projection. */
    public static final class Part {
        private final String name;
        private final int outputSize;
        private final float[] weights;

        public Part(String name, int outputSize, float[] weights) {
            this.name = Objects.requireNonNull(name, "name");
            if (outputSize < 1) {
                throw new IllegalArgumentException("outputSize must be positive: " + outputSize);
            }
            this.outputSize = outputSize;
            this.weights = Objects.requireNonNull(weights, "weights");
        }

        public String name() {
            return name;
        }

        public int outputSize() {
            return outputSize;
        }
    }
}
