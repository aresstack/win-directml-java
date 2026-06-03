package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Pre-defined HLSL compute shaders for the Qwen2 V2.0 GPU pipeline.
 *
 * <p>Analogous to {@link Phi3ComputeShaders} but adapted for Qwen2 architecture:
 * <ul>
 *   <li>element-add: unchanged (same as Phi-3)</li>
 *   <li>RMSNorm: unchanged (same as Phi-3)</li>
 *   <li>SwiGLU: <b>no activation-scale</b> — Qwen2 does not use per-projection
 *       output scales that Phi-3 AWQ quantization requires. The activation is
 *       <code>silu(gate) * up = gate * sigmoid(gate) * up</code>.</li>
 * </ul>
 *
 * <p>Used by {@code QwenGpuPipeline} for the MLP-batch submission that collapses
 * o_proj + residual-add + RMSNorm + gate_up + SwiGLU + down_proj + residual-add
 * into a single D3D12 command list, halving the number of fence waits per token
 * (96 → 48 submissions for 24-layer Qwen 0.5B).
 */
public final class QwenComputeShaders {

    private static final Logger log = LoggerFactory.getLogger(QwenComputeShaders.class);

    private QwenComputeShaders() {
    }

    /**
     * Thread group size for all element-wise shaders.
     */
    public static final int GROUP_SIZE = 256;

    // ═══════════════════════════════════════════════════════════════════
    // Element-wise Add: C[i] = A[i] + B[i]
    // Root params: u0=A, u1=B, u2=C, b0={count}
    // Same HLSL as Phi3ComputeShaders — reused verbatim.
    // ═══════════════════════════════════════════════════════════════════
    public static final String ELEMENT_ADD_HLSL = Phi3ComputeShaders.ELEMENT_ADD_HLSL;

    // ═══════════════════════════════════════════════════════════════════
    // RMSNorm: out[i] = x[i] * weight[i] * rsqrt(mean(x²) + eps)
    // Root params: u0=Input, u1=Weight, u2=Output, b0={dim, eps_bits}
    // Same HLSL as Phi3ComputeShaders — reused verbatim.
    // ═══════════════════════════════════════════════════════════════════
    public static final String RMSNORM_HLSL = Phi3ComputeShaders.RMSNORM_HLSL;

    // ═══════════════════════════════════════════════════════════════════
    // SwiGLU (no activation scale): out[i] = gate[i] * sigmoid(gate[i]) * up[i]
    // Input layout: GateUp = [gate_0..gate_{N-1} | up_0..up_{N-1}]
    // Root params: u0=GateUp, u1=Output, b0={intermediate}
    // ═══════════════════════════════════════════════════════════════════
    public static final String SWIGLU_HLSL = """
            RWByteAddressBuffer GateUp : register(u0);
            RWByteAddressBuffer Output : register(u1);
            cbuffer CB : register(b0) { uint intermediate; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint i = dtid.x;
                if (i < intermediate) {
                    float gate = asfloat(GateUp.Load(i * 4));
                    float up   = asfloat(GateUp.Load((intermediate + i) * 4));
                    float sigmoid = 1.0f / (1.0f + exp(-gate));
                    Output.Store(i * 4, asuint(up * gate * sigmoid));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // ARGMAX: find index of max element in a float array
    // Root params: u0=Input (logits), u1=Output (uint), b0={vocabSize}
    // One thread group: 256 threads cooperatively find max + index
    // ═══════════════════════════════════════════════════════════════════
    public static final String ARGMAX_HLSL = """
            RWByteAddressBuffer Input  : register(u0);
            RWByteAddressBuffer Output : register(u1);
            cbuffer CB : register(b0) { uint vocabSize; };
            
            groupshared float gsMaxVal[256];
            groupshared uint  gsMaxIdx[256];
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 gtid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
                uint t = gtid.x;
                uint localVocab = (vocabSize + 255) / 256;
                float myMax = -1e30f;
                uint  myIdx = 0;
                for (uint i = 0; i < localVocab; i++) {
                    uint idx = t * localVocab + i;
                    if (idx < vocabSize) {
                        float v = asfloat(Input.Load(idx * 4));
                        if (v > myMax) { myMax = v; myIdx = idx; }
                    }
                }
                gsMaxVal[t] = myMax;
                gsMaxIdx[t] = myIdx;
                GroupMemoryBarrierWithGroupSync();
                // Tree reduce
                if (t < 128) { if (gsMaxVal[t+128] > gsMaxVal[t]) { gsMaxVal[t] = gsMaxVal[t+128]; gsMaxIdx[t] = gsMaxIdx[t+128]; } }
                GroupMemoryBarrierWithGroupSync();
                if (t <  64) { if (gsMaxVal[t+64] > gsMaxVal[t]) { gsMaxVal[t] = gsMaxVal[t+64]; gsMaxIdx[t] = gsMaxIdx[t+64]; } }
                GroupMemoryBarrierWithGroupSync();
                if (t <  32) { if (gsMaxVal[t+32] > gsMaxVal[t]) { gsMaxVal[t] = gsMaxVal[t+32]; gsMaxIdx[t] = gsMaxIdx[t+32]; } }
                GroupMemoryBarrierWithGroupSync();
                if (t <  16) { if (gsMaxVal[t+16] > gsMaxVal[t]) { gsMaxVal[t] = gsMaxVal[t+16]; gsMaxIdx[t] = gsMaxIdx[t+16]; } }
                GroupMemoryBarrierWithGroupSync();
                if (t <   8) { if (gsMaxVal[t+8] > gsMaxVal[t]) { gsMaxVal[t] = gsMaxVal[t+8]; gsMaxIdx[t] = gsMaxIdx[t+8]; } }
                GroupMemoryBarrierWithGroupSync();
                if (t <   4) { if (gsMaxVal[t+4] > gsMaxVal[t]) { gsMaxVal[t] = gsMaxVal[t+4]; gsMaxIdx[t] = gsMaxIdx[t+4]; } }
                GroupMemoryBarrierWithGroupSync();
                if (t <   2) { if (gsMaxVal[t+2] > gsMaxVal[t]) { gsMaxVal[t] = gsMaxVal[t+2]; gsMaxIdx[t] = gsMaxIdx[t+2]; } }
                GroupMemoryBarrierWithGroupSync();
                if (t == 0) {
                    if (gsMaxVal[1] > gsMaxVal[0]) gsMaxIdx[0] = gsMaxIdx[1];
                    Output.Store(0, gsMaxIdx[0]);
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // Factory
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Compile all compute kernels needed for the Qwen2 V2.0 GPU pipeline MLP batch.
     *
     * @param wb      initialised WindowsBindings
     * @param cmdList the command list from {@link GpuPipeline#getCommandList()}
     * @throws WindowsNativeException if HLSL compilation fails
     */
    public static ComputeKernelSet createAll(WindowsBindings wb,
                                             java.lang.foreign.MemorySegment cmdList)
            throws WindowsNativeException {
        long t0 = System.currentTimeMillis();

        GpuComputeKernel addKernel = new GpuComputeKernel(wb, cmdList,
                ELEMENT_ADD_HLSL, "qwen_element_add", 3, 1, GROUP_SIZE);

        GpuComputeKernel rmsNormKernel = new GpuComputeKernel(wb, cmdList,
                RMSNORM_HLSL, "qwen_rms_norm", 3, 2, GROUP_SIZE);

        // 2 UAV roots (GateUp + Output), no Scale buffer
        GpuComputeKernel swigluKernel = new GpuComputeKernel(wb, cmdList,
                SWIGLU_HLSL, "qwen_swiglu", 2, 1, GROUP_SIZE);

        // Argmax: 2 UAV (Input, Output uint), 1 constant (vocabSize)
        GpuComputeKernel argmaxKernel = new GpuComputeKernel(wb, cmdList,
                ARGMAX_HLSL, "qwen_argmax", 2, 1, 256);

        log.info("Qwen compute shaders compiled in {} ms",
                System.currentTimeMillis() - t0);
        return new ComputeKernelSet(addKernel, rmsNormKernel, swigluKernel, argmaxKernel);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Kernel set
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Bundle of all compute kernels needed for Qwen2 V2.0 MLP batch.
     */
    public record ComputeKernelSet(
            GpuComputeKernel add,
            GpuComputeKernel rmsNorm,
            GpuComputeKernel swiglu,
            GpuComputeKernel argmax
    ) implements AutoCloseable {
        @Override
        public void close() {
            add.close();
            rmsNorm.close();
            swiglu.close();
            argmax.close();
        }
    }
}
