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

        log.info("Qwen compute shaders compiled in {} ms",
                System.currentTimeMillis() - t0);
        return new ComputeKernelSet(addKernel, rmsNormKernel, swigluKernel);
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
            GpuComputeKernel swiglu
    ) implements AutoCloseable {
        @Override
        public void close() {
            add.close();
            rmsNorm.close();
            swiglu.close();
        }
    }
}
