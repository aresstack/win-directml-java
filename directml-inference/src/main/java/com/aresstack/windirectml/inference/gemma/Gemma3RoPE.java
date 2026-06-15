package com.aresstack.windirectml.inference.gemma;

/**
 * Device-free Gemma 3 rotary position embedding (RoPE) helper — the parity oracle the WARP RoPE kernel
 * mirrors (GEMMA-WARP-7).
 *
 * <p>Gemma uses GPT-NeoX style <b>rotate-half</b> RoPE over each {@code head_dim}-wide head, with a
 * <b>per-layer base frequency</b> (dual theta: global {@code 1e6} on full-attention layers, local
 * {@code 1e4} on sliding-window layers — see {@link Gemma3AttentionLayout#ropeTheta(int)}). The single
 * head primitive is {@link Gemma3ReferenceMath#applyRopeHalf}; this class applies it across all heads of
 * a packed {@code [numHeads * head_dim]} q/k vector. {@code head_dim} is supplied by the caller (Gemma's
 * 256 is decoupled from {@code hidden/heads}); it is never derived as {@code hidden / heads}.</p>
 */
public final class Gemma3RoPE {

    private Gemma3RoPE() {
    }

    /**
     * Apply rotate-half RoPE in place to every head of {@code packed} ({@code numHeads * headDim}
     * values, head {@code h} at offset {@code h * headDim}) for sequence position {@code pos} with base
     * {@code theta}.
     */
    public static void applyToHeads(float[] packed, int numHeads, int headDim, int pos, double theta) {
        if (numHeads < 1 || headDim < 1) {
            throw new IllegalArgumentException("numHeads and headDim must be positive: numHeads="
                    + numHeads + ", headDim=" + headDim);
        }
        if (packed.length != (long) numHeads * headDim) {
            throw new IllegalArgumentException("packed length must equal numHeads*headDim: packed="
                    + packed.length + ", expected=" + ((long) numHeads * headDim));
        }
        for (int head = 0; head < numHeads; head++) {
            Gemma3ReferenceMath.applyRopeHalf(packed, head * headDim, headDim, pos, theta);
        }
    }
}
