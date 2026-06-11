package com.aresstack.windirectml.inference.decoderonly;

import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.GpuPipeline;
import com.aresstack.windirectml.windows.MatMulNBitsKernel;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * GPU-resident MLP block for the decoder-only decode path: {@code gate_up → WARP SwiGLU → down} executed in a single
 * D3D12 command-list submission, with the only CPU readback happening after {@code down}.
 *
 * <p>Replaces the previous per-projection round-trips
 * ({@code gate_up GEMM → readback gate/up → CPU SwiGLU → upload intermediate → down GEMM → readback}) which cost two
 * fence waits and a readback+re-upload of the MLP intermediate per layer per token. Here the gate/up output and the
 * SwiGLU intermediate stay in GPU buffers; only the final {@code down} output is read back.</p>
 *
 * <p>The math is unchanged (same fused {@code [gate|up]} projection, same {@code silu(gate)*up}, same {@code down}
 * projection), so the result is numerically identical to the separate path apart from GPU floating-point rounding.
 * This is the proven Qwen {@code batchMlp} pattern, generalised for the shared decoder-only path.</p>
 */
public final class DecoderOnlyWarpMlpBlock {

    private final GpuPipeline pipeline;
    private final MatMulNBitsKernel gateUpKernel;
    private final MatMulNBitsKernel downKernel;
    private final DecoderOnlyWarpSwiGluKernel swiGlu;
    private final int hiddenSize;
    private final long hiddenBytes;

    private final MemorySegment uavBarrier;
    private final MemorySegment downOutUavToCopySource;

    /**
     * @param pipeline     shared GPU pipeline (one command list + fence reused across layers/steps)
     * @param gateUpKernel the fused gate/up projection kernel (output is {@code [gate | up]}, width {@code 2*inter})
     * @param downKernel   the down projection kernel (input {@code [inter]}, output {@code [hidden]})
     * @param swiGlu       the shared WARP SwiGLU kernel
     * @param hiddenSize   hidden width (= down output width)
     */
    public DecoderOnlyWarpMlpBlock(GpuPipeline pipeline,
                                   MatMulNBitsKernel gateUpKernel,
                                   MatMulNBitsKernel downKernel,
                                   DecoderOnlyWarpSwiGluKernel swiGlu,
                                   int hiddenSize) {
        this.pipeline = Objects.requireNonNull(pipeline, "pipeline");
        this.gateUpKernel = Objects.requireNonNull(gateUpKernel, "gateUpKernel");
        this.downKernel = Objects.requireNonNull(downKernel, "downKernel");
        this.swiGlu = Objects.requireNonNull(swiGlu, "swiGlu");
        if (hiddenSize < 1) {
            throw new IllegalArgumentException("hiddenSize must be positive: " + hiddenSize);
        }
        this.hiddenSize = hiddenSize;
        this.hiddenBytes = (long) hiddenSize * Float.BYTES;
        this.uavBarrier = pipeline.allocUavBarrier();
        this.downOutUavToCopySource = pipeline.allocTransitionBarrier(
                downKernel.getOutputBuf(),
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);
    }

    /**
     * Run the MLP block for a single decode token.
     *
     * @param mlpNormInput the post-attention-RMSNorm hidden state [hidden] (CPU)
     * @param out          destination for the down projection result [hidden] (CPU)
     */
    public void project(float[] mlpNormInput, float[] out) {
        Objects.requireNonNull(mlpNormInput, "mlpNormInput");
        Objects.requireNonNull(out, "out");
        if (out.length < hiddenSize) {
            throw new IllegalArgumentException("out too small: " + out.length + " < " + hiddenSize);
        }

        pipeline.begin();
        MemorySegment cl = pipeline.getCommandList();

        // 1. gate_up: upload mlpNorm → gateUp.inputBuf, GEMM → gateUp.outputBuf (stays UAV).
        gateUpKernel.recordBatchFromCpu(pipeline, mlpNormInput);
        pipeline.recordUavBarrier(uavBarrier);

        // 2. SwiGLU: silu(gate)*up over gateUp.outputBuf → down.inputBuf (stays UAV).
        swiGlu.recordDispatch(cl, gateUpKernel.getOutputBuf(), downKernel.getInputBuf());
        pipeline.recordUavBarrier(uavBarrier);

        // 3. down: GEMM from down.inputBuf (written by SwiGLU) → down.outputBuf (stays UAV).
        downKernel.recordBatchDispatchOnly(pipeline);

        // 4. Read back only the down output.
        pipeline.recordBarrier(downOutUavToCopySource);
        pipeline.recordReadback(downKernel.getOutputBuf(), 0, hiddenBytes);
        pipeline.submitAndWait();
        pipeline.readbackInto(out, 0, hiddenSize);
    }

    public int hiddenSize() {
        return hiddenSize;
    }
}
