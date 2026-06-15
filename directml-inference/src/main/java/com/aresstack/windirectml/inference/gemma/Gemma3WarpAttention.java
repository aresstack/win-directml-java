package com.aresstack.windirectml.inference.gemma;

/**
 * Gemma 3-specific HLSL shaders for the WARP/DirectML attention primitives (GEMMA-WARP-7b).
 *
 * <p>Two device-mirrors of the verified CPU reference attention math
 * ({@code Gemma3ReferenceForwardPass} + {@link Gemma3AttentionLayout}):</p>
 * <ul>
 *   <li><b>RoPE</b> — GPT-NeoX rotate-half over each {@code head_dim}-wide head, per-layer base theta
 *       (dual local/global). One thread per {@code (head, i)} pair, rotating {@code (i, i+half)}.</li>
 *   <li><b>Attention scores</b> — scaled {@code QK^T} for one query position over a packed key buffer,
 *       with GQA head→kv mapping and the sliding-window/causal visible range {@code [firstValid,
 *       queryPos]} supplied by {@link Gemma3AttentionLayout}; out-of-range keys get a large-negative
 *       sentinel so a later softmax ignores them. Layout (full vs local) is decided in Java and fed via
 *       {@code firstValid}, so the kernel stays Gemma-agnostic and never imitates the Qwen path.</li>
 * </ul>
 */
public final class Gemma3WarpAttention {

    private Gemma3WarpAttention() {
    }

    public static final int GROUP_SIZE = 256;

    /** Masked-out score sentinel (matches a softmax that drops these positions). */
    public static final float SCORE_SENTINEL = -1e30f;

    // ═══════════════════════════════════════════════════════════════════
    // RoPE (rotate-half), per head over head_dim:
    //   freq    = theta^-(2i/head_dim),  angle = pos * freq
    //   out[i]      = x[i]*cos - x[i+half]*sin
    //   out[i+half] = x[i+half]*cos + x[i]*sin
    //   Input layout: [numHeads * headDim]; Root params: u0=In, u1=Out,
    //   b0={numHeads, headDim, pos, theta_bits}. One thread per (head, i<half).
    // ═══════════════════════════════════════════════════════════════════
    public static final String ROPE_HLSL = """
            RWByteAddressBuffer In  : register(u0);
            RWByteAddressBuffer Out : register(u1);
            cbuffer CB : register(b0) { uint numHeads; uint headDim; uint pos; uint theta_bits; };

            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint tid = dtid.x;
                uint half = headDim / 2;
                uint total = numHeads * half;
                if (tid >= total) {
                    return;
                }
                uint head = tid / half;
                uint i = tid % half;
                uint base = head * headDim;

                float theta = asfloat(theta_bits);
                float freq = pow(theta, -(2.0f * (float)i) / (float)headDim);
                float angle = (float)pos * freq;
                float c = cos(angle);
                float s = sin(angle);

                float x1 = asfloat(In.Load((base + i) * 4));
                float x2 = asfloat(In.Load((base + i + half) * 4));
                Out.Store((base + i) * 4,        asuint(x1 * c - x2 * s));
                Out.Store((base + i + half) * 4, asuint(x2 * c + x1 * s));
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // Attention scores for one query position:
    //   scores[head, j] = scale * dot(Q[head], K[j][kvHead])   for firstValid <= j <= queryPos
    //                   = SENTINEL                              otherwise
    //   Q: [numHeads*headDim]; K: [seqLen*kvDim] (key j head kvHead at j*kvDim + kvHead*headDim)
    //   Scores: [numHeads*seqLen]; kvHead = head / groupsPerKv
    //   Root params: u0=Q, u1=K, u2=Scores,
    //   b0={numHeads, groupsPerKv, headDim, kvDim, seqLen, queryPos, firstValid, scale_bits}.
    //   One thread per (head, j).
    // ═══════════════════════════════════════════════════════════════════
    public static final String ATTENTION_SCORES_HLSL = """
            RWByteAddressBuffer Q      : register(u0);
            RWByteAddressBuffer K      : register(u1);
            RWByteAddressBuffer Scores : register(u2);
            cbuffer CB : register(b0) {
                uint numHeads; uint groupsPerKv; uint headDim; uint kvDim;
                uint seqLen; uint queryPos; uint firstValid; uint scale_bits;
            };

            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint tid = dtid.x;
                uint total = numHeads * seqLen;
                if (tid >= total) {
                    return;
                }
                uint head = tid / seqLen;
                uint j = tid % seqLen;

                if (j < firstValid || j > queryPos) {
                    Scores.Store(tid * 4, asuint(-1e30f));
                    return;
                }
                uint kvHead = head / groupsPerKv;
                uint qBase = head * headDim;
                uint kBase = j * kvDim + kvHead * headDim;
                float scale = asfloat(scale_bits);
                float acc = 0.0f;
                for (uint c = 0; c < headDim; c++) {
                    acc += asfloat(Q.Load((qBase + c) * 4)) * asfloat(K.Load((kBase + c) * 4));
                }
                Scores.Store(tid * 4, asuint(acc * scale));
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // Row-wise softmax over a [numRows * cols] score matrix (one group per row):
    //   max-subtract, exp, normalize. Masked entries (SENTINEL) exp to 0.
    //   Root params: u0=In, u1=Out, b0={numRows, cols}. Dispatch numRows groups.
    // ═══════════════════════════════════════════════════════════════════
    public static final String SOFTMAX_HLSL = """
            RWByteAddressBuffer In  : register(u0);
            RWByteAddressBuffer Out : register(u1);
            cbuffer CB : register(b0) { uint numRows; uint cols; };

            groupshared float gs[256];

            [numthreads(256, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
                uint row = gid.x;
                if (row >= numRows) {
                    return;
                }
                uint base = row * cols;

                // row max
                float m = -1e30f;
                for (uint i = gi; i < cols; i += 256) {
                    m = max(m, asfloat(In.Load((base + i) * 4)));
                }
                gs[gi] = m;
                GroupMemoryBarrierWithGroupSync();
                for (uint st = 128; st > 0; st >>= 1) {
                    if (gi < st) {
                        gs[gi] = max(gs[gi], gs[gi + st]);
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
                float rowMax = gs[0];
                GroupMemoryBarrierWithGroupSync();

                // sum of exp
                float sum = 0.0f;
                for (uint i = gi; i < cols; i += 256) {
                    sum += exp(asfloat(In.Load((base + i) * 4)) - rowMax);
                }
                gs[gi] = sum;
                GroupMemoryBarrierWithGroupSync();
                for (uint st = 128; st > 0; st >>= 1) {
                    if (gi < st) {
                        gs[gi] += gs[gi + st];
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
                float inv = 1.0f / gs[0];

                for (uint i = gi; i < cols; i += 256) {
                    float e = exp(asfloat(In.Load((base + i) * 4)) - rowMax);
                    Out.Store((base + i) * 4, asuint(e * inv));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // Attention value aggregation: context[head, c] = sum_j prob[head, j] * V[j][kvHead][c]
    //   Prob: [numHeads*cols]; V: [cols*kvDim] (V[j] head kv at j*kvDim + kv*headDim)
    //   Ctx: [numHeads*headDim]; kvHead = head / groupsPerKv
    //   Root params: u0=Prob, u1=V, u2=Ctx, b0={numHeads, groupsPerKv, headDim, kvDim, cols}.
    //   One thread per (head, c).
    // ═══════════════════════════════════════════════════════════════════
    public static final String ATTENTION_VALUE_HLSL = """
            RWByteAddressBuffer Prob : register(u0);
            RWByteAddressBuffer V    : register(u1);
            RWByteAddressBuffer Ctx  : register(u2);
            cbuffer CB : register(b0) {
                uint numHeads; uint groupsPerKv; uint headDim; uint kvDim; uint cols;
            };

            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint tid = dtid.x;
                uint total = numHeads * headDim;
                if (tid >= total) {
                    return;
                }
                uint head = tid / headDim;
                uint c = tid % headDim;
                uint kvHead = head / groupsPerKv;
                float acc = 0.0f;
                for (uint j = 0; j < cols; j++) {
                    float p = asfloat(Prob.Load((head * cols + j) * 4));
                    float vv = asfloat(V.Load((j * kvDim + kvHead * headDim + c) * 4));
                    acc += p * vv;
                }
                Ctx.Store(tid * 4, asuint(acc));
            }
            """;
}
