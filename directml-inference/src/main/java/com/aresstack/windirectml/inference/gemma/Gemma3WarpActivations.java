package com.aresstack.windirectml.inference.gemma;

/**
 * Gemma 3-specific HLSL activation shaders for the WARP/DirectML MLP building blocks (GEMMA-WARP-6).
 *
 * <p>Gemma's MLP is a <b>GeGLU</b> with the <b>GELU-tanh</b> approximation (gelu_pytorch_tanh) — it is
 * <b>not</b> the Qwen/SmolLM2 SwiGLU/SiLU ({@code silu(gate)*up}). These shaders are therefore kept
 * separate from the decoder-only {@code SwiGlu} kernels:</p>
 * <pre>
 *   Gemma:     out_i = gelu_tanh(gate_i) * up_i
 *              gelu_tanh(x) = 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) ))
 *   Qwen/Smol: out_i = silu(gate_i)     * up_i,  silu(x) = x * sigmoid(x)
 * </pre>
 *
 * <p>The GPU mirror of {@link Gemma3ReferenceMath#geluTanh} / {@link Gemma3ReferenceMath#multiplyInPlace}.
 * The constant {@code C = sqrt(2/pi)} matches {@code Gemma3ReferenceMath.GELU_TANH_C}.</p>
 */
public final class Gemma3WarpActivations {

    private Gemma3WarpActivations() {
    }

    /** Thread group size for the element-wise activation shaders. */
    public static final int GROUP_SIZE = 256;

    /** {@code sqrt(2/pi)} — the GELU-tanh coefficient (matches the CPU reference). */
    private static final String C = "0.7978845608028654f";

    // ═══════════════════════════════════════════════════════════════════
    // GELU-tanh, element-wise: out[i] = 0.5 x (1 + tanh(C (x + 0.044715 x^3)))
    //   Root params: u0=Input, u1=Output, b0={count}
    // ═══════════════════════════════════════════════════════════════════
    public static final String GELU_TANH_HLSL = """
            RWByteAddressBuffer Input  : register(u0);
            RWByteAddressBuffer Output : register(u1);
            cbuffer CB : register(b0) { uint count; };

            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint i = dtid.x;
                if (i < count) {
                    float x = asfloat(Input.Load(i * 4));
                    float inner = %C% * (x + 0.044715f * x * x * x);
                    float g = 0.5f * x * (1.0f + tanh(inner));
                    Output.Store(i * 4, asuint(g));
                }
            }
            """.replace("%C%", C);

    // ═══════════════════════════════════════════════════════════════════
    // GeGLU (fused): out[i] = gelu_tanh(gate[i]) * up[i]
    //   Input layout: GateUp = [gate_0..gate_{N-1} | up_0..up_{N-1}]
    //   Root params: u0=GateUp, u1=Output, b0={intermediate}
    // ═══════════════════════════════════════════════════════════════════
    public static final String GEGLU_HLSL = """
            RWByteAddressBuffer GateUp : register(u0);
            RWByteAddressBuffer Output : register(u1);
            cbuffer CB : register(b0) { uint intermediate; };

            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint i = dtid.x;
                if (i < intermediate) {
                    float gate = asfloat(GateUp.Load(i * 4));
                    float up   = asfloat(GateUp.Load((intermediate + i) * 4));
                    float inner = %C% * (gate + 0.044715f * gate * gate * gate);
                    float gelu = 0.5f * gate * (1.0f + tanh(inner));
                    Output.Store(i * 4, asuint(gelu * up));
                }
            }
            """.replace("%C%", C);
}
