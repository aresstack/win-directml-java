package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyReferenceDenseOps;

/**
 * Compatibility facade for SmolLM2 reference dense math.
 *
 * <p>The implementation lives in {@link DecoderOnlyReferenceDenseOps} so Qwen, SmolLM2 and later decoder-only
 * families do not grow duplicate SIMD/scalar dense paths.</p>
 */
final class SmolLM2ReferenceDenseOps {

    private SmolLM2ReferenceDenseOps() {
    }

    static boolean enabled() {
        return DecoderOnlyReferenceDenseOps.enabled();
    }

    static boolean parallelEnabled() {
        return DecoderOnlyReferenceDenseOps.parallelEnabled();
    }

    static float dot(float[] left, int leftOffset, float[] right, int rightOffset, int length) {
        return DecoderOnlyReferenceDenseOps.dot(left, leftOffset, right, rightOffset, length);
    }

    static void multiplyRows(float[] matrix, int rows, int cols, float[] input, float[] output) {
        DecoderOnlyReferenceDenseOps.multiplyRows(matrix, rows, cols, input, output);
    }

    static void gatedSiluMultiply(float[] gate, float[] up) {
        DecoderOnlyReferenceDenseOps.gatedSiluMultiply(gate, up);
    }
}
