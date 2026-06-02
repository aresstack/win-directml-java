package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Pre-defined HLSL compute shaders for the Phi-3 V2.0 GPU pipeline.
 * <p>
 * Each shader operates on UAV root descriptors (no descriptor tables)
 * and is dispatched via {@link GpuComputeKernel}.
 * <p>
 * V2.0 shaders: element-add, RMSNorm, SwiGLU, scale — used for MLP batch
 * (7 GPU ops per layer in 1 submission, 65 submissions/token total).
 */
public final class Phi3ComputeShaders {

    private static final Logger log = LoggerFactory.getLogger(Phi3ComputeShaders.class);

    private Phi3ComputeShaders() {
    }

    /**
     * Thread group size for all element-wise shaders.
     */
    public static final int GROUP_SIZE = 256;

    // ═══════════════════════════════════════════════════════════════════
    // Element-wise Add: C[i] = A[i] + B[i]
    // Root params: u0=A, u1=B, u2=C, b0={count}
    // ═══════════════════════════════════════════════════════════════════

    public static final String ELEMENT_ADD_HLSL = """
            RWByteAddressBuffer A : register(u0);
            RWByteAddressBuffer B : register(u1);
            RWByteAddressBuffer C : register(u2);
            cbuffer CB : register(b0) { uint count; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                if (dtid.x < count) {
                    uint addr = dtid.x * 4;
                    C.Store(addr, asuint(asfloat(A.Load(addr)) + asfloat(B.Load(addr))));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // RMSNorm: out[i] = x[i] * weight[i] * rsqrt(mean(x²) + eps)
    // Root params: u0=Input, u1=Weight, u2=Output, b0={dim, eps_bits}
    // ═══════════════════════════════════════════════════════════════════

    public static final String RMSNORM_HLSL = """
            RWByteAddressBuffer Input  : register(u0);
            RWByteAddressBuffer Weight : register(u1);
            RWByteAddressBuffer Output : register(u2);
            cbuffer CB : register(b0) { uint dim; uint eps_bits; };
            
            groupshared float gs_sum[256];
            
            // Single-group RMSNorm: entire vector processed by one group.
            // Requires dim <= 256 * elements_per_thread.
            // For hidden=3072: each thread handles ceil(3072/256) = 12 elements.
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
                float eps = asfloat(eps_bits);
            
                // Phase 1: Compute partial sum of squares
                float partial = 0.0;
                for (uint i = gi; i < dim; i += 256) {
                    float v = asfloat(Input.Load(i * 4));
                    partial += v * v;
                }
                gs_sum[gi] = partial;
                GroupMemoryBarrierWithGroupSync();
            
                // Phase 2: Tree reduction
                for (uint stride = 128; stride > 0; stride >>= 1) {
                    if (gi < stride) {
                        gs_sum[gi] += gs_sum[gi + stride];
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
            
                // Phase 3: Normalize — rms_inv = rsqrt(sum_sq / dim + eps)
                float rms_inv = rsqrt(gs_sum[0] / (float)dim + eps);
            
                // Phase 4: Apply normalization + weight
                for (uint i = gi; i < dim; i += 256) {
                    float v = asfloat(Input.Load(i * 4));
                    float w = asfloat(Weight.Load(i * 4));
                    Output.Store(i * 4, asuint(v * rms_inv * w));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // SwiGLU + Scale: out[i] = gateUp[i+inter] * gateUp[i] * sigmoid(gateUp[i]) * scale[i]
    // Root params: u0=GateUp, u1=Scale, u2=Output, b0={intermediate}
    // ═══════════════════════════════════════════════════════════════════

    public static final String SWIGLU_HLSL = """
            RWByteAddressBuffer GateUp : register(u0);
            RWByteAddressBuffer Scale  : register(u1);
            RWByteAddressBuffer Output : register(u2);
            cbuffer CB : register(b0) { uint intermediate; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint i = dtid.x;
                if (i < intermediate) {
                    float gate = asfloat(GateUp.Load(i * 4));
                    float up   = asfloat(GateUp.Load((intermediate + i) * 4));
                    float sc   = asfloat(Scale.Load(i * 4));
                    float sigmoid = 1.0 / (1.0 + exp(-gate));
                    Output.Store(i * 4, asuint(up * gate * sigmoid * sc));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // Scale: out[i] = x[i] * scale[i]
    // Root params: u0=X, u1=Scale, u2=Output, b0={count}
    // ═══════════════════════════════════════════════════════════════════

    public static final String SCALE_HLSL = """
            RWByteAddressBuffer X     : register(u0);
            RWByteAddressBuffer Scale : register(u1);
            RWByteAddressBuffer Out   : register(u2);
            cbuffer CB : register(b0) { uint count; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                if (dtid.x < count) {
                    uint addr = dtid.x * 4;
                    Out.Store(addr, asuint(asfloat(X.Load(addr)) * asfloat(Scale.Load(addr))));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // Factory
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Create all compute kernels needed for the Phi-3 V2.0 GPU pipeline.
     */
    public static ComputeKernelSet createAll(WindowsBindings wb, java.lang.foreign.MemorySegment cmdList)
            throws WindowsNativeException {
        long t0 = System.currentTimeMillis();

        GpuComputeKernel addKernel = new GpuComputeKernel(wb, cmdList,
                ELEMENT_ADD_HLSL, "element_add", 3, 1, GROUP_SIZE);
        GpuComputeKernel rmsNormKernel = new GpuComputeKernel(wb, cmdList,
                RMSNORM_HLSL, "rms_norm", 3, 2, GROUP_SIZE);
        GpuComputeKernel swigluKernel = new GpuComputeKernel(wb, cmdList,
                SWIGLU_HLSL, "swiglu", 3, 1, GROUP_SIZE);
        GpuComputeKernel scaleKernel = new GpuComputeKernel(wb, cmdList,
                SCALE_HLSL, "scale", 3, 1, GROUP_SIZE);

        log.info("Phi3 compute shaders compiled in {} ms", System.currentTimeMillis() - t0);
        return new ComputeKernelSet(addKernel, rmsNormKernel, swigluKernel, scaleKernel);
    }

    /**
     * Bundle of all compute kernels for Phi-3 V2.0 MLP batch.
     */
    public record ComputeKernelSet(
            GpuComputeKernel add,
            GpuComputeKernel rmsNorm,
            GpuComputeKernel swiglu,
            GpuComputeKernel scale
    ) implements AutoCloseable {

        @Override
        public void close() {
            add.close();
            rmsNorm.close();
            swiglu.close();
            scale.close();
        }
    }
}
