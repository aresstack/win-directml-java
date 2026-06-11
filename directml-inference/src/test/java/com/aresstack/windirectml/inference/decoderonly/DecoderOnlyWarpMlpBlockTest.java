package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.windows.GpuPipeline;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Numerical verification for the GPU-resident decoder-only MLP block and its WARP SwiGLU kernel.
 *
 * <p>The block runs {@code gate_up → WARP SwiGLU → down} in a single submission with the gate/up output and the
 * SwiGLU intermediate kept GPU-resident (no CPU readback between the two GEMMs). This must be numerically identical
 * (apart from GPU floating-point rounding) to the separate CPU path
 * {@code down(silu(gate) * up)} using {@link DecoderOnlyReferenceDenseOps}.</p>
 */
class DecoderOnlyWarpMlpBlockTest {

    private static final float TOLERANCE = 2.0e-3f;

    @Test
    void mlpBlockMatchesCpuReference() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        int hidden = 8;
        int inter = 16;
        float[] gateWeights = weights(1, inter * hidden);
        float[] upWeights = weights(2, inter * hidden);
        float[] downWeights = weights(3, hidden * inter);
        float[] input = weights(10, hidden);

        // CPU reference: down(silu(gate) * up).
        float[] gateRef = DecoderOnlyReferenceDenseOps.multiplyRows(gateWeights, inter, hidden, input);
        float[] upRef = DecoderOnlyReferenceDenseOps.multiplyRows(upWeights, inter, hidden, input);
        DecoderOnlyReferenceDenseOps.gatedSiluMultiply(gateRef, upRef); // gateRef := silu(gate) * up
        float[] downRef = DecoderOnlyReferenceDenseOps.multiplyRows(downWeights, hidden, inter, gateRef);

        try (WindowsBindings bindings = new WindowsBindings()) {
            bindings.init("warp");
            try (DecoderOnlyWarpFusedDenseProjection gateUp = DecoderOnlyWarpFusedDenseProjection.fromRowMajorParts(
                    bindings, "gate_up", hidden, List.of(
                            new DecoderOnlyWarpFusedDenseProjection.Part("gate", inter, gateWeights),
                            new DecoderOnlyWarpFusedDenseProjection.Part("up", inter, upWeights)));
                 DecoderOnlyWarpDenseProjection down = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(
                         bindings, "down", hidden, inter, downWeights);
                 GpuPipeline pipeline = new GpuPipeline(
                         bindings, (long) hidden * Float.BYTES, (long) hidden * Float.BYTES);
                 DecoderOnlyWarpSwiGluKernel swiGlu = new DecoderOnlyWarpSwiGluKernel(
                         bindings, pipeline.getCommandList(), inter)) {

                DecoderOnlyWarpMlpBlock block = new DecoderOnlyWarpMlpBlock(
                        pipeline, gateUp.kernel(), down.kernel(), swiGlu, hidden);

                float[] out = new float[hidden];
                block.project(input, out);

                assertArrayEquals(downRef, out, TOLERANCE,
                        "GPU-resident MLP block must match the CPU down(silu(gate)*up) reference");
            }
        }
    }

    @Test
    void warpSwiGluMatchesJavaSwiGlu() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        // Identity gate_up and down projections turn the MLP block into a pure SwiGLU probe: with gate_up = identity
        // (input duplicated into [gate | up]) and down = identity, out == silu(gate) * up over the input halves.
        int inter = 16;
        int hidden = inter; // down: [inter] → [inter] identity
        float[] downIdentity = identity(inter);                    // inter x inter identity
        float[] input = weights(77, hidden);

        // Java reference SwiGLU over (gate=input, up=input).
        float[] gate = input.clone();
        float[] up = input.clone();
        DecoderOnlyReferenceDenseOps.gatedSiluMultiply(gate, up); // gate := silu(input) * input

        try (WindowsBindings bindings = new WindowsBindings()) {
            bindings.init("warp");
            try (DecoderOnlyWarpFusedDenseProjection gateUp = DecoderOnlyWarpFusedDenseProjection.fromRowMajorParts(
                    bindings, "gate_up_id", hidden, List.of(
                            new DecoderOnlyWarpFusedDenseProjection.Part("gate", inter, identity(hidden)),
                            new DecoderOnlyWarpFusedDenseProjection.Part("up", inter, identity(hidden))));
                 DecoderOnlyWarpDenseProjection down = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(
                         bindings, "down_id", inter, inter, downIdentity);
                 GpuPipeline pipeline = new GpuPipeline(
                         bindings, (long) hidden * Float.BYTES, (long) hidden * Float.BYTES);
                 DecoderOnlyWarpSwiGluKernel swiGlu = new DecoderOnlyWarpSwiGluKernel(
                         bindings, pipeline.getCommandList(), inter)) {

                DecoderOnlyWarpMlpBlock block = new DecoderOnlyWarpMlpBlock(
                        pipeline, gateUp.kernel(), down.kernel(), swiGlu, inter);

                float[] out = new float[inter];
                block.project(input, out);

                assertArrayEquals(gate, out, TOLERANCE,
                        "WARP SwiGLU must match the Java silu(gate)*up reference");
            }
        }
    }

    /** Row-major identity matrix of size {@code n x n}. */
    private static float[] identity(int n) {
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) {
            m[i * n + i] = 1.0f;
        }
        return m;
    }

    /** Deterministic small pseudo-random weights in roughly [-0.5, 0.5]. */
    private static float[] weights(int seed, int count) {
        float[] values = new float[count];
        long state = seed * 2654435761L + 1442695040888963407L;
        for (int i = 0; i < count; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            int bits = (int) (state >>> 40);
            values[i] = ((bits & 0xFFFF) / 65535.0f) - 0.5f;
        }
        return values;
    }
}
