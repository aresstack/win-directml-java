package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Verifies that the fused QKV / Gate-Up projection is numerically identical to running the parts separately —
 * stacking weight matrices vertically must change only the dispatch count, not the math.
 */
class DecoderOnlyWarpFusedDenseProjectionTest {

    private static final float TOLERANCE = 1.0e-3f;

    @Test
    void fusedQkvMatchesSeparateProjections() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        int inputSize = 5;
        int qOut = 6;
        int kOut = 3;
        int vOut = 3;
        float[] qWeights = weights(1, qOut * inputSize);
        float[] kWeights = weights(2, kOut * inputSize);
        float[] vWeights = weights(3, vOut * inputSize);
        float[] input = weights(10, inputSize);

        try (WindowsBindings bindings = new WindowsBindings()) {
            bindings.init("warp");
            try (DecoderOnlyWarpDenseProjection qRef = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(bindings, "q", qOut, inputSize, qWeights);
                 DecoderOnlyWarpDenseProjection kRef = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(bindings, "k", kOut, inputSize, kWeights);
                 DecoderOnlyWarpDenseProjection vRef = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(bindings, "v", vOut, inputSize, vWeights);
                 DecoderOnlyWarpFusedDenseProjection fused = DecoderOnlyWarpFusedDenseProjection.fromRowMajorParts(
                         bindings, "qkv", inputSize, List.of(
                                 new DecoderOnlyWarpFusedDenseProjection.Part("q", qOut, qWeights),
                                 new DecoderOnlyWarpFusedDenseProjection.Part("k", kOut, kWeights),
                                 new DecoderOnlyWarpFusedDenseProjection.Part("v", vOut, vWeights)))) {

                // Single-vector path.
                float[] fullOutput = fused.project(input);
                float[] q = new float[qOut];
                float[] k = new float[kOut];
                float[] v = new float[vOut];
                fused.copySlice(fullOutput, 0, q);
                fused.copySlice(fullOutput, 1, k);
                fused.copySlice(fullOutput, 2, v);
                assertArrayEquals(qRef.project(input), q, TOLERANCE, "fused q slice");
                assertArrayEquals(kRef.project(input), k, TOLERANCE, "fused k slice");
                assertArrayEquals(vRef.project(input), v, TOLERANCE, "fused v slice");

                // Batched sequence path.
                int seq = 3;
                float[] inputSeq = weights(20, seq * inputSize);
                float[] fusedSeq = new float[seq * fused.totalOutputSize()];
                fused.projectSequenceInto(inputSeq, seq, fusedSeq);
                float[] qSeq = new float[seq * qOut];
                float[] kSeq = new float[seq * kOut];
                float[] vSeq = new float[seq * vOut];
                fused.copySliceSequence(fusedSeq, seq, 0, qSeq);
                fused.copySliceSequence(fusedSeq, seq, 1, kSeq);
                fused.copySliceSequence(fusedSeq, seq, 2, vSeq);
                float[] qRefSeq = new float[seq * qOut];
                float[] kRefSeq = new float[seq * kOut];
                float[] vRefSeq = new float[seq * vOut];
                qRef.projectSequenceInto(inputSeq, seq, qRefSeq);
                kRef.projectSequenceInto(inputSeq, seq, kRefSeq);
                vRef.projectSequenceInto(inputSeq, seq, vRefSeq);
                assertArrayEquals(qRefSeq, qSeq, TOLERANCE, "fused q sequence slice");
                assertArrayEquals(kRefSeq, kSeq, TOLERANCE, "fused k sequence slice");
                assertArrayEquals(vRefSeq, vSeq, TOLERANCE, "fused v sequence slice");
            }
        }
    }

    @Test
    void fusedGateUpMatchesSeparateProjections() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows + D3D12/DirectML");
        int inputSize = 4;
        int interSize = 7;
        float[] gateWeights = weights(31, interSize * inputSize);
        float[] upWeights = weights(32, interSize * inputSize);
        float[] input = weights(40, inputSize);

        try (WindowsBindings bindings = new WindowsBindings()) {
            bindings.init("warp");
            try (DecoderOnlyWarpDenseProjection gateRef = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(bindings, "gate", interSize, inputSize, gateWeights);
                 DecoderOnlyWarpDenseProjection upRef = DecoderOnlyWarpDenseProjection.fromRowMajorWeights(bindings, "up", interSize, inputSize, upWeights);
                 DecoderOnlyWarpFusedDenseProjection fused = DecoderOnlyWarpFusedDenseProjection.fromRowMajorParts(
                         bindings, "gate_up", inputSize, List.of(
                                 new DecoderOnlyWarpFusedDenseProjection.Part("gate", interSize, gateWeights),
                                 new DecoderOnlyWarpFusedDenseProjection.Part("up", interSize, upWeights)))) {

                float[] fullOutput = fused.project(input);
                float[] gate = new float[interSize];
                float[] up = new float[interSize];
                fused.copySlice(fullOutput, 0, gate);
                fused.copySlice(fullOutput, 1, up);
                assertArrayEquals(gateRef.project(input), gate, TOLERANCE, "fused gate slice");
                assertArrayEquals(upRef.project(input), up, TOLERANCE, "fused up slice");
            }
        }
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
