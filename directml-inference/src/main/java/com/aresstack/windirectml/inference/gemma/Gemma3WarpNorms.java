package com.aresstack.windirectml.inference.gemma;

/**
 * Gemma 3-specific HLSL compute shaders for the WARP/DirectML norm building blocks (GEMMA-WARP-5a).
 *
 * <p>These are <b>distinct from</b> the Phi-3/Qwen {@code RMSNORM_HLSL}: Gemma uses a
 * <b>zero-centered</b> RMSNorm — it scales by {@code (1 + weight)}, not {@code weight}:</p>
 * <pre>
 *   Gemma:     y_i = x_i * rsqrt(mean(x^2) + eps) * (1 + w_i)
 *   Qwen/Smol: y_i = x_i * rsqrt(mean(x^2) + eps) *      w_i
 * </pre>
 *
 * <p>Both shaders are single-group-per-vector reductions (mirrors the Phi-3 RMSNorm shape): one D3D12
 * thread group of {@link #GROUP_SIZE} threads cooperatively reduces the sum of squares with a strided
 * load + tree reduction, so any {@code dim} (Gemma hidden=640, head_dim=256) is handled by one group.
 * They are the GPU mirror of {@link Gemma3ReferenceMath#rmsNormZeroCentered}, validated against that
 * already-verified CPU reference.</p>
 */
public final class Gemma3WarpNorms {

    private Gemma3WarpNorms() {
    }

    /** Thread group size for the norm reductions. */
    public static final int GROUP_SIZE = 256;

    // ═══════════════════════════════════════════════════════════════════
    // Zero-centered RMSNorm over a whole vector (e.g. the hidden=640 state).
    //   out[i] = x[i] * rsqrt(mean(x^2) + eps) * (1 + weight[i])
    //   Root params: u0=Input, u1=Weight, u2=Output, b0={dim, eps_bits}
    //   Dispatch: exactly ONE group (elementCount <= GROUP_SIZE).
    // ═══════════════════════════════════════════════════════════════════
    public static final String ZERO_CENTERED_RMSNORM_HLSL = """
            RWByteAddressBuffer Input  : register(u0);
            RWByteAddressBuffer Weight : register(u1);
            RWByteAddressBuffer Output : register(u2);
            cbuffer CB : register(b0) { uint dim; uint eps_bits; };

            groupshared float gs_sum[256];

            [numthreads(256, 1, 1)]
            void CSMain(uint gi : SV_GroupIndex) {
                float eps = asfloat(eps_bits);

                float partial = 0.0;
                for (uint i = gi; i < dim; i += 256) {
                    float v = asfloat(Input.Load(i * 4));
                    partial += v * v;
                }
                gs_sum[gi] = partial;
                GroupMemoryBarrierWithGroupSync();

                for (uint stride = 128; stride > 0; stride >>= 1) {
                    if (gi < stride) {
                        gs_sum[gi] += gs_sum[gi + stride];
                    }
                    GroupMemoryBarrierWithGroupSync();
                }

                float rms_inv = rsqrt(gs_sum[0] / (float)dim + eps);

                // Gemma is zero-centered: scale by (1 + weight), not weight.
                for (uint i = gi; i < dim; i += 256) {
                    float v = asfloat(Input.Load(i * 4));
                    float w = asfloat(Weight.Load(i * 4));
                    Output.Store(i * 4, asuint(v * rms_inv * (1.0f + w)));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // QK-Norm: zero-centered RMSNorm applied PER HEAD over head_dim.
    //   Input layout: [numHeads * headDim] (head h at offset h*headDim)
    //   Weight: [headDim], shared across all heads (Gemma's q_norm / k_norm)
    //   out[h*headDim + i] = x[..] * rsqrt(mean_head(x^2) + eps) * (1 + weight[i])
    //   Root params: u0=Input, u1=Weight, u2=Output, b0={numHeads, headDim, eps_bits}
    //   Dispatch: numHeads groups (group g normalizes head g).
    // ═══════════════════════════════════════════════════════════════════
    public static final String QK_NORM_HLSL = """
            RWByteAddressBuffer Input  : register(u0);
            RWByteAddressBuffer Weight : register(u1);
            RWByteAddressBuffer Output : register(u2);
            cbuffer CB : register(b0) { uint numHeads; uint headDim; uint eps_bits; };

            groupshared float gs_sum[256];

            [numthreads(256, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
                uint head = gid.x;
                if (head >= numHeads) {
                    return;
                }
                uint base = head * headDim;
                float eps = asfloat(eps_bits);

                float partial = 0.0;
                for (uint i = gi; i < headDim; i += 256) {
                    float v = asfloat(Input.Load((base + i) * 4));
                    partial += v * v;
                }
                gs_sum[gi] = partial;
                GroupMemoryBarrierWithGroupSync();

                for (uint stride = 128; stride > 0; stride >>= 1) {
                    if (gi < stride) {
                        gs_sum[gi] += gs_sum[gi + stride];
                    }
                    GroupMemoryBarrierWithGroupSync();
                }

                float rms_inv = rsqrt(gs_sum[0] / (float)headDim + eps);

                for (uint i = gi; i < headDim; i += 256) {
                    float v = asfloat(Input.Load((base + i) * 4));
                    float w = asfloat(Weight.Load(i * 4));
                    Output.Store((base + i) * 4, asuint(v * rms_inv * (1.0f + w)));
                }
            }
            """;
}
