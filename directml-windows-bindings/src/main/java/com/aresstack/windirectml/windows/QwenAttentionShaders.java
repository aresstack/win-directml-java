package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;

/**
 * HLSL compute shaders for GPU-resident Qwen2 attention (Opt-B).
 *
 * <p>Two shaders that together replace the per-token CPU attention block in
 * {@code Qwen2Runtime.processLayerDecode} (~907 ms/token on Intel iGPU):
 *
 * <ol>
 *   <li>{@link #ROPE_AND_APPEND_HLSL} — applies GPT-J-style RoPE to Q and K,
 *       writes rotated K and untouched V into the KV cache at position
 *       {@code pos}. Q is written in-place into the Q buffer.</li>
 *   <li>{@link #GQA_ATTENTION_DECODE_HLSL} — single-token GQA attention with
 *       online-softmax (FlashAttention-style): one thread group per query
 *       head, {@code headDim} threads cooperate over {@code seqLen} positions,
 *       running max/sum maintained in groupshared memory.</li>
 * </ol>
 *
 * <p>Numerically equivalent to the CPU reference
 * ({@code Qwen2Runtime.processLayerDecode} Z. 572-614).
 * Online softmax is mathematically identical to two-pass softmax; floating-
 * point sums may differ at the last bit because addition order changes.
 *
 * <h2>Memory layout assumptions</h2>
 * <ul>
 *   <li>{@code Q}: {@code [qSize]} = {@code [numHeads * headDim]}, head-major
 *       row-major.</li>
 *   <li>{@code K}, {@code V}: {@code [kvSize]} = {@code [kvHeads * headDim]},
 *       kvHead-major row-major.</li>
 *   <li>{@code KCache}, {@code VCache}: {@code [kvHeads * maxSeqLen * headDim]}
 *       kvHead → pos → d. Tail positions past current {@code pos} are
 *       garbage; the attention shader never reads past {@code seqLen}.</li>
 *   <li>{@code AttnOut}: {@code [qSize]}, head-major row-major.</li>
 * </ul>
 *
 * <h2>Dispatch geometry</h2>
 * <table>
 *   <tr><th>Shader</th><th>Groups</th><th>Threads/group</th><th>Thread mapping</th></tr>
 *   <tr>
 *     <td>rope_and_append</td>
 *     <td>numHeads + kvHeads</td>
 *     <td>halfDim (= headDim/2)</td>
 *     <td>thread t = rotation pair index (rotates (x[t], x[halfDim+t]))</td>
 *   </tr>
 *   <tr>
 *     <td>gqa_attention_decode</td>
 *     <td>numHeads</td>
 *     <td>headDim</td>
 *     <td>thread t = dimension index in attnOut[h]</td>
 *   </tr>
 * </table>
 *
 * <p>For Qwen2.5-Coder-0.5B (numHeads=14, kvHeads=2, headDim=64):
 * RoPE dispatches 16 groups × 32 threads; attention dispatches 14 groups × 64 threads.
 */
public final class QwenAttentionShaders {

    private static final Logger log = LoggerFactory.getLogger(QwenAttentionShaders.class);

    private QwenAttentionShaders() {
    }

    // ═══════════════════════════════════════════════════════════════════
    // RoPE + KV-Append (combined)
    // Root params:
    //   u0=Q          (numHeads * headDim floats, in/out — rotated in place)
    //   u1=K          (kvHeads  * headDim floats, in — rotated K is NOT written back)
    //   u2=V          (kvHeads  * headDim floats, in)
    //   u3=KCache     (kvHeads * maxSeqLen * headDim floats, out — write rotated K at pos)
    //   u4=VCache     (kvHeads * maxSeqLen * headDim floats, out — write V at pos)
    //   b0={numHeads, kvHeads, headDim, pos, maxSeqLen, ropeThetaBits}
    // Dispatch: (numHeads + kvHeads) groups × halfDim threads
    //   gid.x in [0..numHeads)            → Q head h, rotate Q[h*headDim .. ]
    //   gid.x in [numHeads..numHeads+kvHeads) → kv head kvH, rotate K[kvH*headDim..]
    //                                             and write KCache + VCache
    // ═══════════════════════════════════════════════════════════════════
    public static final String ROPE_AND_APPEND_HLSL = """
            RWByteAddressBuffer Q      : register(u0);
            RWByteAddressBuffer K      : register(u1);
            RWByteAddressBuffer V      : register(u2);
            RWByteAddressBuffer KCache : register(u3);
            RWByteAddressBuffer VCache : register(u4);
            cbuffer CB : register(b0) {
                uint numHeads;
                uint kvHeads;
                uint headDim;
                uint pos;
                uint maxSeqLen;
                uint ropeThetaBits;
            };
            
            [numthreads(32, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
                uint headIdx = gid.x;
                uint t       = ltid.x;
                uint halfDim = headDim / 2;
                if (t >= halfDim) return;
            
                // Compute cos/sin for this position+dim-pair.
                float theta = asfloat(ropeThetaBits);
                float freq  = 1.0 / pow(theta, (2.0 * float(t)) / float(headDim));
                float angle = float(pos) * freq;
                float c, s;
                sincos(angle, s, c);
            
                if (headIdx < numHeads) {
                    // Q head: rotate in place
                    uint qOff = headIdx * headDim;
                    float x0 = asfloat(Q.Load((qOff + t)           * 4));
                    float x1 = asfloat(Q.Load((qOff + halfDim + t) * 4));
                    Q.Store((qOff + t)           * 4, asuint(x0 * c - x1 * s));
                    Q.Store((qOff + halfDim + t) * 4, asuint(x0 * s + x1 * c));
                } else {
                    // KV head: rotate K and write rotated K + V to cache at pos
                    uint kvH    = headIdx - numHeads;
                    uint kvOff  = kvH * headDim;
                    uint cacheBase = (kvH * maxSeqLen + pos) * headDim;
                    float k0 = asfloat(K.Load((kvOff + t)           * 4));
                    float k1 = asfloat(K.Load((kvOff + halfDim + t) * 4));
                    float v0 = asfloat(V.Load((kvOff + t)           * 4));
                    float v1 = asfloat(V.Load((kvOff + halfDim + t) * 4));
                    float rk0 = k0 * c - k1 * s;
                    float rk1 = k0 * s + k1 * c;
                    KCache.Store((cacheBase + t)           * 4, asuint(rk0));
                    KCache.Store((cacheBase + halfDim + t) * 4, asuint(rk1));
                    VCache.Store((cacheBase + t)           * 4, asuint(v0));
                    VCache.Store((cacheBase + halfDim + t) * 4, asuint(v1));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // GQA Attention Decode (single Q, online softmax)
    //
    // Root params:
    //   u0=Q        (numHeads * headDim floats, in)
    //   u1=KCache   (kvHeads * maxSeqLen * headDim floats, in)
    //   u2=VCache   (kvHeads * maxSeqLen * headDim floats, in)
    //   u3=AttnOut  (numHeads * headDim floats, out)
    //   b0={numHeads, kvHeads, headDim, qHeadsPerKvHead, seqLen, maxSeqLen, scaleBits}
    //
    // Dispatch: numHeads groups × headDim threads.
    //   Each group computes attnOut[h*headDim + t] for one query head h.
    //   Per position p in [0..seqLen):
    //     1) Each thread loads K[kvH][p][t], computes Q[h][t] * K (partial dot).
    //     2) Tree-reduce partial dots in groupshared → full dot product.
    //     3) Online softmax update on thread 0:
    //          score  = dot * scale
    //          m_new  = max(m, score)
    //          alpha  = exp(m - m_new)
    //          e      = exp(score - m_new)
    //          s      = s * alpha + e
    //          m      = m_new
    //     4) Each thread updates its outVal:
    //          outVal = outVal * alpha + e * V[kvH][p][t]
    //   Final: attnOut[h][t] = outVal / s
    //
    // Numerical equivalence to CPU two-pass softmax: the running-max
    // formulation is the standard FlashAttention recurrence and is
    // mathematically identical; floating-point sum order differs in the
    // last bit only.
    // ═══════════════════════════════════════════════════════════════════
    public static final String GQA_ATTENTION_DECODE_HLSL = """
            RWByteAddressBuffer Q       : register(u0);
            RWByteAddressBuffer KCache  : register(u1);
            RWByteAddressBuffer VCache  : register(u2);
            RWByteAddressBuffer AttnOut : register(u3);
            cbuffer CB : register(b0) {
                uint numHeads;
                uint kvHeads;
                uint headDim;
                uint qHeadsPerKvHead;
                uint seqLen;
                uint maxSeqLen;
                uint scaleBits;
            };
            
            #define MAX_HEAD_DIM 128
            groupshared float qShared[MAX_HEAD_DIM];
            groupshared float partialDots[MAX_HEAD_DIM];
            groupshared float gsM;
            groupshared float gsS;
            groupshared float gsAlpha;
            groupshared float gsE;
            
            [numthreads(64, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
                uint h = gid.x;
                uint t = ltid.x;
                // NOTE: no early-return on (t >= headDim) — it would put the
                // following GroupMemoryBarrierWithGroupSync calls into varying
                // flow control (HLSL error X4026). Instead, all 64 threads run
                // every loop iteration; threads outside [0..headDim) contribute
                // 0 to partial dots and skip the final store.
                bool active = (t < headDim);
            
                float scale = asfloat(scaleBits);
                uint kvH = h / qHeadsPerKvHead;
                uint qOff = h * headDim + t;
                uint kvCacheBase = kvH * maxSeqLen * headDim;
            
                // Load Q[h][t] into groupshared (reused across all positions).
                float qVal = active ? asfloat(Q.Load(qOff * 4)) : 0.0f;
                qShared[t] = qVal;
            
                if (t == 0) {
                    gsM = -1e30f;
                    gsS = 0.0f;
                }
                float outVal = 0.0f;
                GroupMemoryBarrierWithGroupSync();
            
                for (uint p = 0; p < seqLen; p++) {
                    uint kvOff = (kvCacheBase + p * headDim + t) * 4;
                    float kVal = active ? asfloat(KCache.Load(kvOff)) : 0.0f;
                    float vVal = active ? asfloat(VCache.Load(kvOff)) : 0.0f;
            
                    // Partial dot: thread t contributes qShared[t] * kVal.
                    // Inactive threads contribute 0 (qShared[t] is 0 for them).
                    partialDots[t] = qShared[t] * kVal;
                    GroupMemoryBarrierWithGroupSync();
            
                    // Tree reduction across all 64 threads (group size).
                    // Barriers sit AFTER the if-blocks so every thread reaches
                    // them (uniform flow control at the barrier point).
                    if (t < 32) partialDots[t] += partialDots[t + 32];
                    GroupMemoryBarrierWithGroupSync();
                    if (t < 16) partialDots[t] += partialDots[t + 16];
                    GroupMemoryBarrierWithGroupSync();
                    if (t <  8) partialDots[t] += partialDots[t + 8];
                    GroupMemoryBarrierWithGroupSync();
                    if (t <  4) partialDots[t] += partialDots[t + 4];
                    GroupMemoryBarrierWithGroupSync();
                    if (t <  2) partialDots[t] += partialDots[t + 2];
                    GroupMemoryBarrierWithGroupSync();
            
                    if (t == 0) {
                        float dotFull = partialDots[0] + partialDots[1];
                        float score   = dotFull * scale;
                        float mNew    = max(gsM, score);
                        gsAlpha       = exp(gsM - mNew);
                        gsE           = exp(score - mNew);
                        gsS           = gsS * gsAlpha + gsE;
                        gsM           = mNew;
                    }
                    GroupMemoryBarrierWithGroupSync();
            
                    // Online softmax output update (all threads read gsAlpha/gsE)
                    outVal = outVal * gsAlpha + gsE * vVal;
                }
            
                // Normalize and store (only active threads write)
                if (active) {
                    float invS = 1.0f / gsS;
                    AttnOut.Store((h * headDim + t) * 4, asuint(outVal * invS));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // Factory
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Compile both attention shaders and return them as a bundle.
     *
     * @param wb      initialised WindowsBindings (D3D12 device required)
     * @param cmdList command list to associate MethodHandle cache with
     * @return compiled and ready-to-dispatch shader set
     */
    public static AttentionKernelSet createAll(WindowsBindings wb, MemorySegment cmdList)
            throws WindowsNativeException {
        long t0 = System.currentTimeMillis();

        // ROPE_AND_APPEND: 5 UAVs (Q, K, V, KCache, VCache),
        //                  6 root constants (numHeads, kvHeads, headDim, pos, maxSeqLen, ropeThetaBits),
        //                  group size = 32 threads (= halfDim for Qwen2.5-Coder 0.5B headDim=64).
        GpuComputeKernel ropeKernel = new GpuComputeKernel(wb, cmdList,
                ROPE_AND_APPEND_HLSL, "qwen_rope_and_append",
                5, 6, 32);

        // GQA_ATTENTION_DECODE: 4 UAVs (Q, KCache, VCache, AttnOut),
        //                       7 root constants (numHeads, kvHeads, headDim,
        //                                          qHeadsPerKvHead, seqLen, maxSeqLen, scaleBits),
        //                       group size = 64 threads (= headDim).
        GpuComputeKernel attnKernel = new GpuComputeKernel(wb, cmdList,
                GQA_ATTENTION_DECODE_HLSL, "qwen_gqa_attention_decode",
                4, 7, 64);

        log.info("Qwen attention shaders compiled in {} ms",
                System.currentTimeMillis() - t0);
        return new AttentionKernelSet(ropeKernel, attnKernel);
    }

    /**
     * Bundle of both attention shaders. The shaders are stateless after
     * compilation; multiple decoder layers share these compiled PSOs and
     * differ only in their per-dispatch UAV addresses and root constants.
     */
    public record AttentionKernelSet(
            GpuComputeKernel ropeAndAppend,
            GpuComputeKernel gqaAttentionDecode
    ) implements AutoCloseable {
        @Override
        public void close() {
            ropeAndAppend.close();
            gqaAttentionDecode.close();
        }
    }
}
