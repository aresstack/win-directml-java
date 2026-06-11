package com.aresstack.windirectml.inference.decoderonly;

/**
 * CPU reference projection seam for the decoder-only WARP forward pass.
 *
 * <p>Only used for the dev/diagnostic LM-head fallback: when a family opts the LM head onto the CPU-SIMD reference
 * path instead of WARP, it supplies the projection through this seam. This is never part of the WARP product path — it
 * exists so the generic forward pass can run the diagnostic LM head without depending on a family tensor type.</p>
 */
@FunctionalInterface
public interface DecoderOnlyReferenceProjection {

    /**
     * Project {@code input} and return a freshly allocated output vector.
     */
    float[] project(float[] input);
}
