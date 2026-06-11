package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.GpuComputeKernel;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * WARP/DirectML SwiGLU activation kernel for decoder-only MLP blocks.
 *
 * <p>Computes {@code out[i] = silu(gate[i]) * up[i] = gate[i] * sigmoid(gate[i]) * up[i]} fully on the GPU, reading a
 * fused {@code [gate_0..gate_{N-1} | up_0..up_{N-1}]} buffer (exactly the layout produced by the fused gate/up
 * projection) and writing the {@code [intermediate]} result into a GPU buffer. This keeps the MLP intermediate
 * GPU-resident, removing the CPU readback of gate/up and the re-upload of the activation between the two GEMMs.</p>
 *
 * <p>The activation matches the Java reference {@code DecoderOnlyReferenceDenseOps.gatedSiluMultiply} apart from GPU
 * floating-point rounding (verified by {@code warpSwiGluMatchesJavaSwiGlu}).</p>
 */
public final class DecoderOnlyWarpSwiGluKernel implements AutoCloseable {

    // silu(gate)*up over a fused [gate | up] buffer. Family-neutral (no activation scale).
    private static final String SWIGLU_HLSL = """
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

    private final GpuComputeKernel kernel;
    private final int intermediate;

    /**
     * @param windowsBindings initialised WARP bindings
     * @param cmdListForMethodHandles a command list used only to cache method handles (typically the shared pipeline's)
     * @param intermediate the MLP intermediate width
     */
    public DecoderOnlyWarpSwiGluKernel(WindowsBindings windowsBindings,
                                       MemorySegment cmdListForMethodHandles,
                                       int intermediate) throws WindowsNativeException {
        Objects.requireNonNull(windowsBindings, "windowsBindings");
        Objects.requireNonNull(cmdListForMethodHandles, "cmdListForMethodHandles");
        if (intermediate < 1) {
            throw new IllegalArgumentException("intermediate must be positive: " + intermediate);
        }
        this.intermediate = intermediate;
        // 2 UAV roots (GateUp, Output) + 1 root constant (intermediate), group size 256.
        this.kernel = new GpuComputeKernel(windowsBindings, cmdListForMethodHandles,
                SWIGLU_HLSL, "decoderonly_swiglu", 2, 1, 256);
    }

    public int intermediate() {
        return intermediate;
    }

    /**
     * Record the SwiGLU dispatch into a command list: reads {@code gateUpBuffer} ([2*intermediate]) and writes the
     * {@code [intermediate]} result into {@code outputBuffer}. Both buffers must be GPU default (UAV) buffers.
     */
    public void recordDispatch(MemorySegment commandList, MemorySegment gateUpBuffer, MemorySegment outputBuffer) {
        kernel.recordDispatch(commandList,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(gateUpBuffer),
                        D3D12Bindings.getGpuVirtualAddress(outputBuffer)
                },
                new int[]{intermediate},
                intermediate);
    }

    @Override
    public void close() {
        kernel.close();
    }
}
