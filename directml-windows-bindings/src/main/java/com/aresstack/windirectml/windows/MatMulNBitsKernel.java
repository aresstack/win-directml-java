package com.aresstack.windirectml.windows;

import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.DxgiBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * GPU-accelerated MatMulNBits kernel for AWQ INT4 block-128 quantized weights.
 * <p>
 * This is the <b>single most important kernel</b> in the Phi-3 driver.
 * Almost every projection in the model (q/k/v/o, gate_up, down, lm_head)
 * depends on this kernel.
 * <p>
 * <b>V1 strategy</b>: Dequantize INT4→FP32 once at preparation time, upload
 * the full FP32 weight matrix to GPU, then use DirectML GEMM for inference.
 * This trades GPU memory for implementation simplicity. A future V2 can
 * replace this with a custom D3D12 compute shader operating directly on
 * packed INT4 data.
 * <p>
 * <b>V1.1 performance optimization</b>: All per-call resources (staging buffers,
 * command allocator, command list, fence, binding table) are pre-allocated at
 * preparation time. The {@link #matvec(float[])} hot path combines upload,
 * DML dispatch, and readback into a <b>single command list submission</b>,
 * reducing GPU synchronization from 3× to 1× per call and eliminating
 * COM object churn entirely.
 * <p>
 * <b>Kernel contract</b>:
 * <ul>
 *   <li>Input:  x ∈ FP32 [M, K]  (M=1 for decode, M=seqLen for prefill)</li>
 *   <li>Weight: packed INT4 uint8 + FP16 scales + uint4 zero-points, block=128</li>
 *   <li>Output: y ∈ FP32 [M, N]  where y = x @ W^T</li>
 * </ul>
 * <p>
 * No ONNX Runtime, no JNI, no JNA. Pure Java 21 FFM → D3D12 → DirectML.
 */
public final class MatMulNBitsKernel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(MatMulNBitsKernel.class);

    private final WindowsBindings wb;
    private final Arena arena;
    private final int N;  // output features (rows of weight matrix)
    private final int K;  // input features  (cols of weight matrix)

    // ── GPU resources (created once at prepare time) ─────────────────────
    private MemorySegment weightBuf;     // GPU default buffer: dequantized FP32 [N, K]
    private MemorySegment biasBuf;       // GPU default buffer: zero-bias [N] (GEMM requires C tensor)
    private MemorySegment inputBuf;      // GPU default buffer: input [1, K] (reused per call)
    private MemorySegment outputBuf;     // GPU default buffer: output [1, N]

    // ── DML compiled operator ────────────────────────────────────────────
    private MemorySegment compiledGemm;
    private MemorySegment descriptorHeap;
    private MemorySegment cmdRecorder;
    private int descriptorIncrement;

    // ── Binding properties ───────────────────────────────────────────────
    private int descCount;
    private long tempSize;
    private long persistSize;
    private MemorySegment tempBuf;
    private MemorySegment persistBuf;

    // ── Pre-allocated per-call resources (V1.1 optimization) ─────────────
    private MemorySegment uploadBuf;        // upload heap: [K] floats, persistently mapped
    private MemorySegment readbackBuf;      // readback heap: [N] floats, persistently mapped
    private MemorySegment mappedUpload;     // persistently mapped CPU pointer for upload
    private MemorySegment mappedReadback;   // persistently mapped CPU pointer for readback
    private MemorySegment execAllocator;    // reusable command allocator
    private MemorySegment execCmdList;      // reusable command list (reset per call)
    private MemorySegment execBindingTable; // reusable DML binding table
    private MemorySegment execFence;        // reusable fence (value increments per call)
    private long fenceValue;                // monotonically increasing fence counter

    // ── Pre-allocated barrier/heap structs for zero-alloc hot path (V1.2) ─
    private MemorySegment barrierInputToUAV;      // transition: COPY_DEST → UAV
    // V1.3: barrierUAV removed — redundant, transition barrier provides sync
    private MemorySegment barrierOutputToCS;       // transition: UAV → COPY_SOURCE
    private MemorySegment barrierInputToCommon;    // transition: UAV → COMMON
    private MemorySegment barrierOutputToCommon;   // transition: COPY_SOURCE → COMMON
    private MemorySegment heapArrayPtr;            // single-element heap array for SetDescriptorHeaps
    private MemorySegment cmdListArrayPtr;         // single-element cmd list array for ExecuteCommandLists

    private boolean prepared = false;
    private boolean closed = false;

    // ── INT4 GPU mode (V3.0 — 8× memory-bandwidth reduction) ─────────────────
    /**
     * Set {@code -Ddirectml.int4.gpu.disabled=true} to force the legacy FP32/DML
     * GEMM path (useful for benchmarking or debugging). Defaults to enabled.
     */
    private static final boolean INT4_GPU_ENABLED =
            !Boolean.getBoolean("directml.int4.gpu.disabled");

    /**
     * Whether this instance uses the INT4 GPU compute path or the legacy DML FP32 path.
     */
    private final boolean useInt4Gpu;
    private int int4BlockSize;             // quantisation block size (e.g., 128)
    private MemorySegment int4WeightBuf;   // GPU default buffer: packed INT4 [N * K/2 bytes]
    private MemorySegment int4ScalesBuf;   // GPU default buffer: FP32 scales [N * blocksPerRow * 4]
    private MemorySegment int4ZpBuf;       // GPU default buffer: packed INT4 zero-points
    private GpuComputeKernel int4Shader;   // compiled HLSL INT4 MatVec kernel
    private final boolean useWarpParallelInt4Matvec;
    private int warpParallelInt4MatvecRows;
    private static final int WARP_PARALLEL_INT4_MATVEC_GROUP_SIZE_ROWS16 = 64;
    private static final int WARP_PARALLEL_INT4_MATVEC_GROUP_SIZE_ROWS32 = 128;
    private static final int WARP_PARALLEL_INT4_MATVEC_ROWS16 = 16;
    private static final int WARP_PARALLEL_INT4_MATVEC_ROWS32 = 32;

    /**
     * ROWS=32 is now opt-in only. The ROWS=32 benchmark did not beat the
     * ROWS=16 kernel on WARP for Qwen lm_head; the default path stays on
     * ROWS=16 and uses a leaner 64-thread group. Override with
     * -Ddirectml.int4.warpLmHeadRows32.minN=... to benchmark ROWS=32 again,
     * or keep directml.int4.warpLmHeadRows32.disabled=true to force ROWS=16.
     */
    private static final int WARP_LMHEAD_ROWS32_MIN_N =
            Integer.getInteger("directml.int4.warpLmHeadRows32.minN", Integer.MAX_VALUE);
    private long[] int4UavAddrs;           // pre-cached GPU VAs: [X, QW, Scales, ZP, Y]
    private int[] int4Constants;           // pre-cached constants: [N, K, blockSize]

    // ── Batched matmul (Opt-A: prefill) — lazily allocated on first matmulBatch call ─────
    // Opt-A-2 v2 (2026-05-29): the batch *shaders* are SHARED across all kernel
    // instances. The HLSL source is identical for every layer × projection, so
    // compiling it 96 times (one per kernel) wasted ~30–40 s of prefill on
    // Intel iGPU per the diagnosis log. Static + lazy init = compile once,
    // reuse for every kernel.
    //
    // Opt-A-3 (2026-06-03): the batch *scratch buffers* are NOW ALSO SHARED.
    // During prefill, layers run sequentially — only ONE matmulBatch() executes
    // at a time — so a single set of scratch buffers can be reused across all
    // 96 kernel instances. This drops VRAM from 96×20MB → 1×20MB.
    //
    // The static shaders AND buffers are intentionally leaked for the lifetime of
    // the JVM (a few KB per PSO + RootSig + buffers) — re-allocating would
    // re-introduce the compile-and-upload cost when a kernel is recreated.
    private static volatile GpuComputeKernel sharedInt4BatchShader;
    private static volatile GpuComputeKernel sharedFp32BatchShader;
    private static final Object batchShaderLock = new Object();

    // ── SHARED batch scratch buffers (static — one set for all kernel instances) ──
    private static volatile MemorySegment sharedInt4BatchInputBuf;    // GPU default [maxBatchM, K]
    private static volatile MemorySegment sharedInt4BatchOutputBuf;   // GPU default [maxBatchM, N]
    private static volatile MemorySegment sharedInt4BatchUploadBuf;   // upload heap [maxBatchM * K * 4 bytes]
    private static volatile MemorySegment sharedInt4BatchReadbackBuf; // readback heap [maxBatchM * N * 4 bytes]
    private static volatile MemorySegment sharedInt4BatchMappedUpload;    // persistently mapped upload
    private static volatile MemorySegment sharedInt4BatchMappedReadback; // persistently mapped readback
    private static volatile int sharedBatchMaxK = 0;               // max K seen across all kernel instances (for shared buffer sizing)
    private static volatile int sharedBatchMaxN = 0;                  // max N seen across all kernel instances
    private static volatile int sharedBatchCapacityM = 0;              // currently-allocated M capacity (static)
    private static final Object batchBufferLock = new Object();

    // ── Static Arena for SHARED batch buffers (lives for JVM lifetime) ────────
    // MUST be separate from per-instance 'arena' — otherwise arena.close() on first
    // kernel.close() invalidates ALL static buffers while other instances still use them.
    private static volatile Arena sharedBatchArena = null;
    private static final Object sharedArenaLock = new Object();

    // ── Device ownership of the SHARED batch resources ────────────────────────
    // The static shaders/buffers above are bound to ONE D3D12 device — the device
    // of whichever model loaded first. They are intentionally retained for the JVM
    // lifetime so a model with many layers compiles the batch shader only once.
    //
    // BUT when the workbench switches models, the previous model's WindowsBindings
    // (and its D3D12 device) is closed. The static resources then dangle on a dead
    // device. Re-using GPU buffers/shaders created on device A from a command queue
    // on device B raises DXGI_ERROR_DEVICE_REMOVED — observed as the T5 "matvec
    // failed" that only happens after another model (Qwen/Phi-3) ran first.
    //
    // We therefore record the owning device address. When a kernel on a DIFFERENT
    // device asks for batch capacity, we release the stale static resources and
    // rebuild them against the new device (see ensureBatchCapacity).
    private static volatile long sharedBatchDeviceAddr = 0L;
    // The JVM shutdown hook must be registered only once, even though the static
    // arena may be re-created when the owning device changes.
    private static volatile boolean batchShutdownHookRegistered = false;

    // ── Per-instance references to the shared buffers (for clean code) ─────────────
    // These are NOT allocated per-instance — they just point to the shared buffers.
    // Kept as instance fields so existing code (int4BatchInputBuf, etc.) still works.
    private MemorySegment int4BatchInputBuf;        // → sharedInt4BatchInputBuf
    private MemorySegment int4BatchOutputBuf;       // → sharedInt4BatchOutputBuf
    private MemorySegment int4BatchUploadBuf;       // → sharedInt4BatchUploadBuf
    private MemorySegment int4BatchReadbackBuf;     // → sharedInt4BatchReadbackBuf
    private MemorySegment int4BatchMappedUpload;    // → sharedInt4BatchMappedUpload
    private MemorySegment int4BatchMappedReadback;  // → sharedInt4BatchMappedReadback
    private int batchCapacityM = 0;                 // → sharedBatchCapacityM
    private MemorySegment barrierBatchInputToUAV;
    private MemorySegment barrierBatchOutputToCS;
    private MemorySegment barrierBatchInputToCommon;
    private MemorySegment barrierBatchOutputToCommon;

    /**
     * Hard upper bound on batched M to keep VRAM bounded (~tens of MB per kernel).
     */
    private static final int MAX_BATCH_M = 256; // reduced for iGPU VRAM (96 kernels × M×K×4 must fit)

    /**
     * Whether this kernel's batch scratch buffers were successfully allocated.
     * Set to {@code false} by {@link #ensureBatchCapacity} on OOM; callers
     * should check {@link #supportsBatch()} and fall back to per-row dispatch.
     */
    private boolean batchEnabled = true;

    // ── HLSL shader source for INT4 block-quantised matrix-vector product ────
    // Registers: u0=X (FP32 input), u1=QW (packed INT4 weights),
    //            u2=Scales (FP32), u3=ZP (packed INT4), u4=Y (FP32 output)
    // Constants (b0): N, K, blockSz
    //
    // One thread handles one output row n. Reads 4 packed INT4 bytes at a time
    // (= 8 nibbles = 8 weights) for efficient 4-byte-aligned VRAM access.
    // Bandwidth: only N*K/2 bytes of weight data vs N*K*4 for FP32 → 8× reduction.
    private static final String INT4_MATVEC_HLSL = """
            RWByteAddressBuffer X      : register(u0);
            RWByteAddressBuffer QW     : register(u1);
            RWByteAddressBuffer Scales : register(u2);
            RWByteAddressBuffer ZP     : register(u3);
            RWByteAddressBuffer Y      : register(u4);
            cbuffer CB : register(b0) { uint N; uint K; uint blockSz; };
            
            [numthreads(64, 1, 1)]
            void CSMain(uint3 tid : SV_DispatchThreadID) {
                uint n = tid.x;
                if (n >= N) return;
            
                uint blocksPerRow = K / blockSz;
                float sum = 0.0;
            
                for (uint blk = 0; blk < blocksPerRow; blk++) {
                    uint scIdx = n * blocksPerRow + blk;
            
                    // FP32 scale for this block
                    float scale = asfloat(Scales.Load(scIdx * 4));
            
                    // Packed uint4 zero-point nibble (2 per byte, low nibble first)
                    uint zpByteIdx  = scIdx / 2;
                    uint zpDword    = ZP.Load((zpByteIdx / 4) * 4);
                    uint zpByte     = (zpDword >> ((zpByteIdx % 4) * 8)) & 0xFF;
                    float zpVal = (scIdx % 2u == 0u) ? float(zpByte & 0xFu)
                                                     : float(zpByte >> 4u);
            
                    // Base byte offset in QW for row n, block blk
                    uint qByteBase = n * blocksPerRow * (blockSz / 2) + blk * (blockSz / 2);
                    // Base k-index for this block
                    uint kBase = blk * blockSz;
            
                    // Process 4 bytes (8 nibbles = 8 weights) per inner iteration.
                    // qByteBase is always 4-aligned (blockSz multiple of 8), so
                    // QW.Load(qByteBase + j) is always 4-byte-aligned.
                    for (uint j = 0; j < blockSz / 2; j += 4) {
                        uint dword = QW.Load(qByteBase + j);
                        uint b0 = dword & 0xFF;
                        uint b1 = (dword >> 8)  & 0xFF;
                        uint b2 = (dword >> 16) & 0xFF;
                        uint b3 =  dword >> 24;
                        uint kOff = kBase + j * 2;
                        sum += (float(b0 & 0xF) - zpVal) * scale * asfloat(X.Load((kOff+0)*4));
                        sum += (float(b0 >> 4)  - zpVal) * scale * asfloat(X.Load((kOff+1)*4));
                        sum += (float(b1 & 0xF) - zpVal) * scale * asfloat(X.Load((kOff+2)*4));
                        sum += (float(b1 >> 4)  - zpVal) * scale * asfloat(X.Load((kOff+3)*4));
                        sum += (float(b2 & 0xF) - zpVal) * scale * asfloat(X.Load((kOff+4)*4));
                        sum += (float(b2 >> 4)  - zpVal) * scale * asfloat(X.Load((kOff+5)*4));
                        sum += (float(b3 & 0xF) - zpVal) * scale * asfloat(X.Load((kOff+6)*4));
                        sum += (float(b3 >> 4)  - zpVal) * scale * asfloat(X.Load((kOff+7)*4));
                    }
                }
            
                Y.Store(n * 4, asuint(sum));
            }
            """;

    // ── WARP-optimised INT4 matrix-vector product ───────────────────────────
    // One thread group computes 16 adjacent output rows. The 128 lanes
    // cooperatively reduce the K dimension and reuse each X element for all
    // 16 rows. This reduces WARP scheduling overhead compared with the
    // one-row-per-group kernel and improves cache locality on CPU execution.
    private static final String INT4_MATVEC_WARP_PARALLEL_HLSL = """
            RWByteAddressBuffer X      : register(u0);
            RWByteAddressBuffer QW     : register(u1);
            RWByteAddressBuffer Scales : register(u2);
            RWByteAddressBuffer ZP     : register(u3);
            RWByteAddressBuffer Y      : register(u4);
            cbuffer CB : register(b0) { uint N; uint K; uint blockSz; };
            
            #define THREADS 64
            #define ROWS 16
            groupshared float partial[THREADS * ROWS];
            groupshared float blockScale[ROWS];
            groupshared float blockZeroPoint[ROWS];
            
            float loadZeroPoint(uint scIdx) {
                uint zpByteIdx = scIdx / 2;
                uint zpDword = ZP.Load((zpByteIdx / 4) * 4);
                uint zpByte = (zpDword >> ((zpByteIdx % 4) * 8)) & 0xFF;
                return (scIdx % 2u == 0u) ? float(zpByte & 0xFu) : float(zpByte >> 4u);
            }
            
            uint loadPackedWeightByte(uint byteAddress) {
                uint dword = QW.Load((byteAddress / 4) * 4);
                return (dword >> ((byteAddress % 4) * 8)) & 0xFF;
            }
            
            [numthreads(THREADS, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
                uint lane = ltid.x;
                uint rowBase = gid.x * ROWS;
                uint blocksPerRow = K / blockSz;
                uint bytesPerBlock = blockSz / 2;
            
                float sum0 = 0.0;
                float sum1 = 0.0;
                float sum2 = 0.0;
                float sum3 = 0.0;
                float sum4 = 0.0;
                float sum5 = 0.0;
                float sum6 = 0.0;
                float sum7 = 0.0;
                float sum8 = 0.0;
                float sum9 = 0.0;
                float sum10 = 0.0;
                float sum11 = 0.0;
                float sum12 = 0.0;
                float sum13 = 0.0;
                float sum14 = 0.0;
                float sum15 = 0.0;
            
                for (uint blk = 0; blk < blocksPerRow; blk++) {
                    if (lane < ROWS) {
                        uint n = rowBase + lane;
                        if (n < N) {
                            uint scIdx = n * blocksPerRow + blk;
                            blockScale[lane] = asfloat(Scales.Load(scIdx * 4));
                            blockZeroPoint[lane] = loadZeroPoint(scIdx);
                        } else {
                            blockScale[lane] = 0.0;
                            blockZeroPoint[lane] = 0.0;
                        }
                    }
                    GroupMemoryBarrierWithGroupSync();
            
                    if (lane < bytesPerBlock) {
                        uint kOff = blk * blockSz + lane * 2;
                        float x0 = asfloat(X.Load((kOff + 0) * 4));
                        float x1 = asfloat(X.Load((kOff + 1) * 4));
            
                        uint n0 = rowBase + 0;
                        if (n0 < N) {
                            uint qByteBase0 = n0 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte0 = loadPackedWeightByte(qByteBase0 + lane);
                            float scale0 = blockScale[0];
                            float zp0 = blockZeroPoint[0];
                            sum0 += (float(qByte0 & 0xFu) - zp0) * scale0 * x0;
                            sum0 += (float(qByte0 >> 4u) - zp0) * scale0 * x1;
                        }
            
                        uint n1 = rowBase + 1;
                        if (n1 < N) {
                            uint qByteBase1 = n1 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte1 = loadPackedWeightByte(qByteBase1 + lane);
                            float scale1 = blockScale[1];
                            float zp1 = blockZeroPoint[1];
                            sum1 += (float(qByte1 & 0xFu) - zp1) * scale1 * x0;
                            sum1 += (float(qByte1 >> 4u) - zp1) * scale1 * x1;
                        }
            
                        uint n2 = rowBase + 2;
                        if (n2 < N) {
                            uint qByteBase2 = n2 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte2 = loadPackedWeightByte(qByteBase2 + lane);
                            float scale2 = blockScale[2];
                            float zp2 = blockZeroPoint[2];
                            sum2 += (float(qByte2 & 0xFu) - zp2) * scale2 * x0;
                            sum2 += (float(qByte2 >> 4u) - zp2) * scale2 * x1;
                        }
            
                        uint n3 = rowBase + 3;
                        if (n3 < N) {
                            uint qByteBase3 = n3 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte3 = loadPackedWeightByte(qByteBase3 + lane);
                            float scale3 = blockScale[3];
                            float zp3 = blockZeroPoint[3];
                            sum3 += (float(qByte3 & 0xFu) - zp3) * scale3 * x0;
                            sum3 += (float(qByte3 >> 4u) - zp3) * scale3 * x1;
                        }
            
                        uint n4 = rowBase + 4;
                        if (n4 < N) {
                            uint qByteBase4 = n4 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte4 = loadPackedWeightByte(qByteBase4 + lane);
                            float scale4 = blockScale[4];
                            float zp4 = blockZeroPoint[4];
                            sum4 += (float(qByte4 & 0xFu) - zp4) * scale4 * x0;
                            sum4 += (float(qByte4 >> 4u) - zp4) * scale4 * x1;
                        }
            
                        uint n5 = rowBase + 5;
                        if (n5 < N) {
                            uint qByteBase5 = n5 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte5 = loadPackedWeightByte(qByteBase5 + lane);
                            float scale5 = blockScale[5];
                            float zp5 = blockZeroPoint[5];
                            sum5 += (float(qByte5 & 0xFu) - zp5) * scale5 * x0;
                            sum5 += (float(qByte5 >> 4u) - zp5) * scale5 * x1;
                        }
            
                        uint n6 = rowBase + 6;
                        if (n6 < N) {
                            uint qByteBase6 = n6 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte6 = loadPackedWeightByte(qByteBase6 + lane);
                            float scale6 = blockScale[6];
                            float zp6 = blockZeroPoint[6];
                            sum6 += (float(qByte6 & 0xFu) - zp6) * scale6 * x0;
                            sum6 += (float(qByte6 >> 4u) - zp6) * scale6 * x1;
                        }
            
                        uint n7 = rowBase + 7;
                        if (n7 < N) {
                            uint qByteBase7 = n7 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte7 = loadPackedWeightByte(qByteBase7 + lane);
                            float scale7 = blockScale[7];
                            float zp7 = blockZeroPoint[7];
                            sum7 += (float(qByte7 & 0xFu) - zp7) * scale7 * x0;
                            sum7 += (float(qByte7 >> 4u) - zp7) * scale7 * x1;
                        }
            
                        uint n8 = rowBase + 8;
                        if (n8 < N) {
                            uint qByteBase8 = n8 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte8 = loadPackedWeightByte(qByteBase8 + lane);
                            float scale8 = blockScale[8];
                            float zp8 = blockZeroPoint[8];
                            sum8 += (float(qByte8 & 0xFu) - zp8) * scale8 * x0;
                            sum8 += (float(qByte8 >> 4u) - zp8) * scale8 * x1;
                        }
            
                        uint n9 = rowBase + 9;
                        if (n9 < N) {
                            uint qByteBase9 = n9 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte9 = loadPackedWeightByte(qByteBase9 + lane);
                            float scale9 = blockScale[9];
                            float zp9 = blockZeroPoint[9];
                            sum9 += (float(qByte9 & 0xFu) - zp9) * scale9 * x0;
                            sum9 += (float(qByte9 >> 4u) - zp9) * scale9 * x1;
                        }
            
                        uint n10 = rowBase + 10;
                        if (n10 < N) {
                            uint qByteBase10 = n10 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte10 = loadPackedWeightByte(qByteBase10 + lane);
                            float scale10 = blockScale[10];
                            float zp10 = blockZeroPoint[10];
                            sum10 += (float(qByte10 & 0xFu) - zp10) * scale10 * x0;
                            sum10 += (float(qByte10 >> 4u) - zp10) * scale10 * x1;
                        }
            
                        uint n11 = rowBase + 11;
                        if (n11 < N) {
                            uint qByteBase11 = n11 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte11 = loadPackedWeightByte(qByteBase11 + lane);
                            float scale11 = blockScale[11];
                            float zp11 = blockZeroPoint[11];
                            sum11 += (float(qByte11 & 0xFu) - zp11) * scale11 * x0;
                            sum11 += (float(qByte11 >> 4u) - zp11) * scale11 * x1;
                        }
            
                        uint n12 = rowBase + 12;
                        if (n12 < N) {
                            uint qByteBase12 = n12 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte12 = loadPackedWeightByte(qByteBase12 + lane);
                            float scale12 = blockScale[12];
                            float zp12 = blockZeroPoint[12];
                            sum12 += (float(qByte12 & 0xFu) - zp12) * scale12 * x0;
                            sum12 += (float(qByte12 >> 4u) - zp12) * scale12 * x1;
                        }
            
                        uint n13 = rowBase + 13;
                        if (n13 < N) {
                            uint qByteBase13 = n13 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte13 = loadPackedWeightByte(qByteBase13 + lane);
                            float scale13 = blockScale[13];
                            float zp13 = blockZeroPoint[13];
                            sum13 += (float(qByte13 & 0xFu) - zp13) * scale13 * x0;
                            sum13 += (float(qByte13 >> 4u) - zp13) * scale13 * x1;
                        }
            
                        uint n14 = rowBase + 14;
                        if (n14 < N) {
                            uint qByteBase14 = n14 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte14 = loadPackedWeightByte(qByteBase14 + lane);
                            float scale14 = blockScale[14];
                            float zp14 = blockZeroPoint[14];
                            sum14 += (float(qByte14 & 0xFu) - zp14) * scale14 * x0;
                            sum14 += (float(qByte14 >> 4u) - zp14) * scale14 * x1;
                        }
            
                        uint n15 = rowBase + 15;
                        if (n15 < N) {
                            uint qByteBase15 = n15 * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                            uint qByte15 = loadPackedWeightByte(qByteBase15 + lane);
                            float scale15 = blockScale[15];
                            float zp15 = blockZeroPoint[15];
                            sum15 += (float(qByte15 & 0xFu) - zp15) * scale15 * x0;
                            sum15 += (float(qByte15 >> 4u) - zp15) * scale15 * x1;
                        }
            
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
            
                partial[lane + THREADS * 0] = sum0;
                partial[lane + THREADS * 1] = sum1;
                partial[lane + THREADS * 2] = sum2;
                partial[lane + THREADS * 3] = sum3;
                partial[lane + THREADS * 4] = sum4;
                partial[lane + THREADS * 5] = sum5;
                partial[lane + THREADS * 6] = sum6;
                partial[lane + THREADS * 7] = sum7;
                partial[lane + THREADS * 8] = sum8;
                partial[lane + THREADS * 9] = sum9;
                partial[lane + THREADS * 10] = sum10;
                partial[lane + THREADS * 11] = sum11;
                partial[lane + THREADS * 12] = sum12;
                partial[lane + THREADS * 13] = sum13;
                partial[lane + THREADS * 14] = sum14;
                partial[lane + THREADS * 15] = sum15;
                GroupMemoryBarrierWithGroupSync();
            
                for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
                    if (lane < stride) {
                        partial[lane + THREADS * 0] += partial[lane + stride + THREADS * 0];
                        partial[lane + THREADS * 1] += partial[lane + stride + THREADS * 1];
                        partial[lane + THREADS * 2] += partial[lane + stride + THREADS * 2];
                        partial[lane + THREADS * 3] += partial[lane + stride + THREADS * 3];
                        partial[lane + THREADS * 4] += partial[lane + stride + THREADS * 4];
                        partial[lane + THREADS * 5] += partial[lane + stride + THREADS * 5];
                        partial[lane + THREADS * 6] += partial[lane + stride + THREADS * 6];
                        partial[lane + THREADS * 7] += partial[lane + stride + THREADS * 7];
                        partial[lane + THREADS * 8] += partial[lane + stride + THREADS * 8];
                        partial[lane + THREADS * 9] += partial[lane + stride + THREADS * 9];
                        partial[lane + THREADS * 10] += partial[lane + stride + THREADS * 10];
                        partial[lane + THREADS * 11] += partial[lane + stride + THREADS * 11];
                        partial[lane + THREADS * 12] += partial[lane + stride + THREADS * 12];
                        partial[lane + THREADS * 13] += partial[lane + stride + THREADS * 13];
                        partial[lane + THREADS * 14] += partial[lane + stride + THREADS * 14];
                        partial[lane + THREADS * 15] += partial[lane + stride + THREADS * 15];
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
            
                if (lane == 0) {
                    uint outRow0 = rowBase + 0;
                    if (outRow0 < N) Y.Store(outRow0 * 4, asuint(partial[THREADS * 0]));
                    uint outRow1 = rowBase + 1;
                    if (outRow1 < N) Y.Store(outRow1 * 4, asuint(partial[THREADS * 1]));
                    uint outRow2 = rowBase + 2;
                    if (outRow2 < N) Y.Store(outRow2 * 4, asuint(partial[THREADS * 2]));
                    uint outRow3 = rowBase + 3;
                    if (outRow3 < N) Y.Store(outRow3 * 4, asuint(partial[THREADS * 3]));
                    uint outRow4 = rowBase + 4;
                    if (outRow4 < N) Y.Store(outRow4 * 4, asuint(partial[THREADS * 4]));
                    uint outRow5 = rowBase + 5;
                    if (outRow5 < N) Y.Store(outRow5 * 4, asuint(partial[THREADS * 5]));
                    uint outRow6 = rowBase + 6;
                    if (outRow6 < N) Y.Store(outRow6 * 4, asuint(partial[THREADS * 6]));
                    uint outRow7 = rowBase + 7;
                    if (outRow7 < N) Y.Store(outRow7 * 4, asuint(partial[THREADS * 7]));
                    uint outRow8 = rowBase + 8;
                    if (outRow8 < N) Y.Store(outRow8 * 4, asuint(partial[THREADS * 8]));
                    uint outRow9 = rowBase + 9;
                    if (outRow9 < N) Y.Store(outRow9 * 4, asuint(partial[THREADS * 9]));
                    uint outRow10 = rowBase + 10;
                    if (outRow10 < N) Y.Store(outRow10 * 4, asuint(partial[THREADS * 10]));
                    uint outRow11 = rowBase + 11;
                    if (outRow11 < N) Y.Store(outRow11 * 4, asuint(partial[THREADS * 11]));
                    uint outRow12 = rowBase + 12;
                    if (outRow12 < N) Y.Store(outRow12 * 4, asuint(partial[THREADS * 12]));
                    uint outRow13 = rowBase + 13;
                    if (outRow13 < N) Y.Store(outRow13 * 4, asuint(partial[THREADS * 13]));
                    uint outRow14 = rowBase + 14;
                    if (outRow14 < N) Y.Store(outRow14 * 4, asuint(partial[THREADS * 14]));
                    uint outRow15 = rowBase + 15;
                    if (outRow15 < N) Y.Store(outRow15 * 4, asuint(partial[THREADS * 15]));
                }
            }
            """;

    // ── WARP-optimised lm_head INT4 matrix-vector product ─────────────────────
    // Same algorithmic idea as ROWS=16, but only selected for very large N
    // (lm_head). It halves the number of WARP thread groups and reuses each
    // loaded X element across 32 vocabulary rows. Decoder layer projections keep
    // ROWS=16 to avoid unnecessary register pressure.
    private static final String INT4_MATVEC_WARP_PARALLEL_ROWS32_HLSL = """
            RWByteAddressBuffer X      : register(u0);
            RWByteAddressBuffer QW     : register(u1);
            RWByteAddressBuffer Scales : register(u2);
            RWByteAddressBuffer ZP     : register(u3);
            RWByteAddressBuffer Y      : register(u4);
            cbuffer CB : register(b0) { uint N; uint K; uint blockSz; };
            
            #define THREADS 128
            #define ROWS 32
            groupshared float partial[THREADS * ROWS];
            groupshared float blockScale[ROWS];
            groupshared float blockZeroPoint[ROWS];
            
            float loadZeroPoint(uint scIdx) {
                uint zpByteIdx = scIdx / 2;
                uint zpDword = ZP.Load((zpByteIdx / 4) * 4);
                uint zpByte = (zpDword >> ((zpByteIdx % 4) * 8)) & 0xFF;
                return (scIdx % 2u == 0u) ? float(zpByte & 0xFu) : float(zpByte >> 4u);
            }
            
            uint loadPackedWeightByte(uint byteAddress) {
                uint dword = QW.Load((byteAddress / 4) * 4);
                return (dword >> ((byteAddress % 4) * 8)) & 0xFF;
            }
            
            [numthreads(THREADS, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
                uint lane = ltid.x;
                uint rowBase = gid.x * ROWS;
                uint blocksPerRow = K / blockSz;
                uint bytesPerBlock = blockSz / 2;
            
                float sum[ROWS];
                [unroll]
                for (uint r = 0; r < ROWS; r++) {
                    sum[r] = 0.0;
                }
            
                for (uint blk = 0; blk < blocksPerRow; blk++) {
                    if (lane < ROWS) {
                        uint n = rowBase + lane;
                        if (n < N) {
                            uint scIdx = n * blocksPerRow + blk;
                            blockScale[lane] = asfloat(Scales.Load(scIdx * 4));
                            blockZeroPoint[lane] = loadZeroPoint(scIdx);
                        } else {
                            blockScale[lane] = 0.0;
                            blockZeroPoint[lane] = 0.0;
                        }
                    }
                    GroupMemoryBarrierWithGroupSync();
            
                    if (lane < bytesPerBlock) {
                        uint kOff = blk * blockSz + lane * 2;
                        float x0 = asfloat(X.Load((kOff + 0) * 4));
                        float x1 = asfloat(X.Load((kOff + 1) * 4));
            
                        [unroll]
                        for (uint r = 0; r < ROWS; r++) {
                            uint n = rowBase + r;
                            if (n < N) {
                                uint qByteBase = n * blocksPerRow * bytesPerBlock + blk * bytesPerBlock;
                                uint qByte = loadPackedWeightByte(qByteBase + lane);
                                float scale = blockScale[r];
                                float zp = blockZeroPoint[r];
                                sum[r] += (float(qByte & 0xFu) - zp) * scale * x0;
                                sum[r] += (float(qByte >> 4u) - zp) * scale * x1;
                            }
                        }
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
            
                [unroll]
                for (uint r = 0; r < ROWS; r++) {
                    partial[lane + THREADS * r] = sum[r];
                }
                GroupMemoryBarrierWithGroupSync();
            
                for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
                    if (lane < stride) {
                        [unroll]
                        for (uint r = 0; r < ROWS; r++) {
                            partial[lane + THREADS * r] += partial[lane + stride + THREADS * r];
                        }
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
            
                if (lane == 0) {
                    [unroll]
                    for (uint r = 0; r < ROWS; r++) {
                        uint outRow = rowBase + r;
                        if (outRow < N) {
                            Y.Store(outRow * 4, asuint(partial[THREADS * r]));
                        }
                    }
                }
            }
            """;

    // ── HLSL shader source for INT4 batched matmul with groupshared X tiling ──
    // Each thread group computes a [TILE, TILE] = [16, 16] output sub-tile.
    // 256 threads cooperatively load a [TILE, TILE_K] = [16, 16] X-sub-tile into
    // groupshared per K-iteration, then each thread accumulates its output cell
    // by dequantizing one INT4 weight row on-the-fly from the compressed buffers.
    //
    // KEY OPTIMISATION: X is loaded cooperatively into groupshared, reducing
    // VRAM traffic for X from M×N reads (naive) to M×(N/16) reads (tiled).
    // For Qwen qkvFused (N=1152, K=896, M=256): X traffic drops from ~264M
    // to ~16.5M fp32 loads — a 16× reduction. Weights stay compressed INT4
    // on the GPU (8× less bandwidth than FP32), and dequantization is done
    // inline per thread with no intermediate FP32 weight storage.
    //
    // Dispatch: 1D, one thread group per output tile. Caller passes
    // elementCount = mTiles * nTiles * 256, groupSize=256.
    // 5 constants: N, K, blockSz, M, nTiles (N-direction tile count).
    private static final String INT4_MATMUL_BATCH_TILED_HLSL = """
            RWByteAddressBuffer X      : register(u0);
            RWByteAddressBuffer QW     : register(u1);
            RWByteAddressBuffer Scales : register(u2);
            RWByteAddressBuffer ZP     : register(u3);
            RWByteAddressBuffer Y      : register(u4);
            cbuffer CB : register(b0) { uint N; uint K; uint blockSz; uint M; uint nTiles; };
            
            #define TILE 16
            #define TILE_K 16
            groupshared float xTile[TILE * TILE_K];
            groupshared float wTile[TILE * TILE_K];
            
            float loadZeroPoint(uint scIdx) {
                uint zpByteIdx = scIdx / 2;
                uint zpDword = ZP.Load((zpByteIdx / 4) * 4);
                uint zpByte = (zpDword >> ((zpByteIdx % 4) * 8)) & 0xFF;
                return (scIdx % 2u == 0u) ? float(zpByte & 0xFu) : float(zpByte >> 4u);
            }
            
            float loadDequantWeight(uint n, uint kAbs, uint blocksPerRow) {
                uint blk = kAbs / blockSz;
                uint kInBlk = kAbs - blk * blockSz;
            
                uint scIdx = n * blocksPerRow + blk;
                float scale = asfloat(Scales.Load(scIdx * 4));
                float zpVal = loadZeroPoint(scIdx);
            
                uint qByteAddr = n * (K / 2) + blk * (blockSz / 2) + kInBlk / 2;
                uint qDword = QW.Load((qByteAddr / 4) * 4);
                uint qByte = (qDword >> ((qByteAddr % 4) * 8)) & 0xFF;
                float qVal = (kInBlk % 2u == 0u)
                        ? float(qByte & 0xFu)
                        : float(qByte >> 4u);
                return (qVal - zpVal) * scale;
            }
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
                uint groupId = gid.x;
                uint gx = groupId % nTiles;
                uint gy = groupId / nTiles;
                uint t  = ltid.x;
                uint tx = t % TILE;
                uint ty = t / TILE;
                uint m  = gy * TILE + ty;
                uint n  = gx * TILE + tx;
            
                uint blocksPerRow = K / blockSz;
                float acc = 0.0;
            
                for (uint kk = 0; kk < K; kk += TILE_K) {
                    // Cooperative load X [TILE_M, TILE_K]. All 256 lanes load one
                    // activation value and the 16 rows reuse this tile.
                    uint xRow = gy * TILE + (t / TILE_K);
                    uint xCol = kk + (t % TILE_K);
                    xTile[(t / TILE_K) * TILE_K + (t % TILE_K)] =
                            (xRow < M && xCol < K)
                                    ? asfloat(X.Load((xRow * K + xCol) * 4))
                                    : 0.0;
            
                    // Cooperative load W [TILE_N, TILE_K]. This is the important
                    // WARP optimisation: every weight is dequantized once per
                    // output tile and reused by up to 16 prefill rows, instead of
                    // every (m,n) thread repeating the same Q4+scale+zp work.
                    uint wRow = gx * TILE + (t / TILE_K);
                    uint wCol = kk + (t % TILE_K);
                    wTile[(t / TILE_K) * TILE_K + (t % TILE_K)] =
                            (wRow < N && wCol < K)
                                    ? loadDequantWeight(wRow, wCol, blocksPerRow)
                                    : 0.0;
            
                    GroupMemoryBarrierWithGroupSync();
            
                    if (m < M && n < N) {
                        [unroll]
                        for (uint ki = 0; ki < TILE_K; ki++) {
                            acc += xTile[ty * TILE_K + ki] * wTile[tx * TILE_K + ki];
                        }
                    }
            
                    GroupMemoryBarrierWithGroupSync();
                }
            
                if (m < M && n < N) {
                    Y.Store((m * N + n) * 4, asuint(acc));
                }
            }
            """;

    // FP32 batched matmul shader for pre-dequantized weights (qkvFused, gateUpFused).
    // Same M-row batching contract as INT4_MATMUL_BATCH_TILED_HLSL; weight buffer is
    // [N, K] row-major FP32 (held in weightBuf for legacy/FP32 mode kernels).
    //
    // Opt-A-2 (2026-05-29): tiled with groupshared memory. Each thread group
    // computes a [TILE, TILE] = [16, 16] output sub-tile. 256 threads per
    // group cooperatively load a [TILE, TILE_K] = [16, 16] X-sub-tile and a
    // [TILE_K, TILE] = [16, 16] W-sub-tile into groupshared per K-iteration,
    // then each thread accumulates TILE_K=16 multiply-adds out of groupshared.
    //
    // HBM traffic per output tile drops from 256 × (2K) = 512K reads (naive)
    // to 32 × K reads (tiled) — ~16× less for K=896 (Qwen 0.5B hidden).
    //
    // Dispatch: 1D, one thread group per output tile. Caller passes
    // elementCount = mTiles * nTiles * 256, groupSize=256 → mTiles*nTiles groups.
    // Constants include nTiles so the shader can decode 1D group-id to (gx, gy).
    private static final String FP32_MATMUL_BATCH_HLSL = """
            RWByteAddressBuffer X : register(u0);
            RWByteAddressBuffer W : register(u1);
            RWByteAddressBuffer Y : register(u2);
            cbuffer CB : register(b0) { uint N; uint K; uint M; uint nTiles; };
            
            #define TILE 16
            #define TILE_K 16
            groupshared float xTile[TILE * TILE_K];
            groupshared float wTile[TILE_K * TILE];
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
                uint groupId = gid.x;
                uint gx = groupId % nTiles;          // tile column (output N-direction)
                uint gy = groupId / nTiles;          // tile row    (output M-direction)
                uint t  = ltid.x;
                uint tx = t % TILE;                  // within-tile column
                uint ty = t / TILE;                  // within-tile row
                uint m  = gy * TILE + ty;
                uint n  = gx * TILE + tx;
            
                float acc = 0.0;
                for (uint kk = 0; kk < K; kk += TILE_K) {
                    // Cooperative load: 256 threads load one X-element each
                    // (16*16 = 256 elements per X-tile) and one W-element each.
                    uint xRow = gy * TILE + (t / TILE_K);
                    uint xCol = kk + (t % TILE_K);
                    uint wRow = kk + (t / TILE);
                    uint wCol = gx * TILE + (t % TILE);
            
                    float xv = (xRow < M && xCol < K)
                            ? asfloat(X.Load((xRow * K + xCol) * 4)) : 0.0;
                    // W is [N, K] row-major — element W[wCol, wRow] at byte (wCol*K + wRow)*4.
                    float wv = (wRow < K && wCol < N)
                            ? asfloat(W.Load((wCol * K + wRow) * 4)) : 0.0;
            
                    xTile[(t / TILE_K) * TILE_K + (t % TILE_K)] = xv;
                    wTile[(t / TILE) * TILE + (t % TILE)] = wv;
            
                    GroupMemoryBarrierWithGroupSync();
            
                    // Each thread accumulates ONE output cell (ty, tx) from the
                    // loaded tiles. 16 multiply-adds per thread per K-iteration.
                    for (uint k = 0; k < TILE_K; k++) {
                        acc += xTile[ty * TILE_K + k] * wTile[k * TILE + tx];
                    }
            
                    GroupMemoryBarrierWithGroupSync();
                }
            
                if (m < M && n < N) {
                    Y.Store((m * N + n) * 4, asuint(acc));
                }
            }
            """;

    /**
     * Create a MatMulNBits kernel for a specific weight matrix.
     *
     * @param wb         initialized WindowsBindings (D3D12 + DirectML devices)
     * @param N          output features (weight rows)
     * @param K          input features (weight cols)
     * @param qWeight    packed INT4 weights [N, K/blockSize, blockSize/2]
     * @param scales     per-block FP32 scales [N * K/blockSize]
     * @param zeroPoints packed uint4 zero points
     * @param blockSize  quantization block size (128)
     */
    public MatMulNBitsKernel(WindowsBindings wb, int N, int K,
                             byte[] qWeight, float[] scales, byte[] zeroPoints,
                             int blockSize) {
        this.wb = wb;
        this.arena = Arena.ofShared();
        this.N = N;
        this.K = K;
        this.useInt4Gpu = INT4_GPU_ENABLED;
        this.useWarpParallelInt4Matvec = useInt4Gpu
                && wb.isWarpBackend()
                && !Boolean.getBoolean("directml.int4.warpParallelMatvec.disabled");
        this.warpParallelInt4MatvecRows = selectWarpParallelRows(N, this.useWarpParallelInt4Matvec);

        try {
            if (useInt4Gpu) {
                prepareInt4Gpu(qWeight, scales, zeroPoints, blockSize);
            } else {
                long t0 = System.nanoTime();
                float[] dequantized = dequantizeInt4(qWeight, scales, zeroPoints, N, K, blockSize);
                log.info("Dequantized [{}, {}] INT4→FP32 in {} ms (legacy DML path)",
                        N, K, (System.nanoTime() - t0) / 1_000_000);
                prepareGpu(dequantized);
            }
        } catch (WindowsNativeException e) {
            arena.close();
            throw new RuntimeException("MatMulNBitsKernel preparation failed", e);
        }
    }

    /**
     * Create a kernel from pre-dequantized FP32 weights.
     * <p>
     * Used for fused projections (e.g., Q+K+V merged into one larger matrix)
     * where the caller has already dequantized and concatenated the weights.
     *
     * @param wb      initialized WindowsBindings
     * @param N       output features (rows)
     * @param K       input features (cols)
     * @param weights pre-dequantized FP32 weight matrix [N * K], row-major
     * @return ready-to-use kernel
     */
    public static MatMulNBitsKernel fromDequantizedWeights(WindowsBindings wb,
                                                           int N, int K,
                                                           float[] weights) {
        return new MatMulNBitsKernel(wb, N, K, weights);
    }

    /**
     * Package-private constructor for pre-dequantized FP32 weights.
     */
    private MatMulNBitsKernel(WindowsBindings wb, int N, int K, float[] fp32Weights) {
        this.wb = wb;
        this.arena = Arena.ofShared();
        this.N = N;
        this.K = K;
        this.useInt4Gpu = false;  // pre-dequantized path always uses DML FP32
        this.useWarpParallelInt4Matvec = false;
        this.warpParallelInt4MatvecRows = 1;

        try {
            prepareGpu(fp32Weights);
        } catch (WindowsNativeException e) {
            arena.close();
            throw new RuntimeException("MatMulNBitsKernel preparation failed (FP32 path)", e);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // INT4 GPU preparation (V3.0 — keeps INT4 on GPU, 8× bandwidth reduction)
    // ══════════════════════════════════════════════════════════════════════

    private static int selectWarpParallelRows(int n, boolean useWarpParallel) {
        if (!useWarpParallel) return 1;
        if (!Boolean.getBoolean("directml.int4.warpLmHeadRows32.disabled")
                && n >= WARP_LMHEAD_ROWS32_MIN_N) {
            return WARP_PARALLEL_INT4_MATVEC_ROWS32;
        }
        return WARP_PARALLEL_INT4_MATVEC_ROWS16;
    }

    private static String warpParallelMatvecShaderSource(int rows) {
        return rows == WARP_PARALLEL_INT4_MATVEC_ROWS32
                ? INT4_MATVEC_WARP_PARALLEL_ROWS32_HLSL
                : INT4_MATVEC_WARP_PARALLEL_HLSL;
    }

    private static int warpParallelInt4MatvecGroupSize(int rows) {
        return rows == WARP_PARALLEL_INT4_MATVEC_ROWS32
                ? WARP_PARALLEL_INT4_MATVEC_GROUP_SIZE_ROWS32
                : WARP_PARALLEL_INT4_MATVEC_GROUP_SIZE_ROWS16;
    }

    /**
     * INT4 GPU path: upload packed INT4 weights + scales + zero-points to GPU,
     * compile the custom HLSL INT4-MatVec shader, and pre-allocate execution
     * infrastructure. Replaces the DML FP32 GEMM path entirely.
     */
    private void prepareInt4Gpu(byte[] qWeight, float[] scales, byte[] zeroPoints,
                                int blockSize) throws WindowsNativeException {
        this.int4BlockSize = blockSize;
        var dev = wb.getD3d12Device();
        var queue = wb.getCommandQueue();

        int blocksPerRow = K / blockSize;
        long qBytes = (long) N * (K / 2);                    // packed INT4
        long scaleBytes = (long) N * blocksPerRow * Float.BYTES;  // FP32 scales
        long zpBytes = ((long) N * blocksPerRow + 1) / 2;      // packed INT4 ZP

        log.info("INT4 GPU mode [{}, {}]: weight={} KB, scales={} KB, zp={} KB (was {} MB FP32)",
                N, K, qBytes / 1024, scaleBytes / 1024, zpBytes / 1024,
                (long) N * K * 4 / (1024 * 1024));

        // Upload INT4 weights to GPU
        int4WeightBuf = D3D12Bindings.createDefaultBuffer(dev, qBytes, arena);
        D3D12Bindings.uploadBytes(dev, queue, int4WeightBuf, qWeight, arena);

        // Upload FP32 scales to GPU (reuse uploadFloats)
        int4ScalesBuf = D3D12Bindings.createDefaultBuffer(dev, scaleBytes, arena);
        D3D12Bindings.uploadFloats(dev, queue, int4ScalesBuf, scales, arena);

        // Upload INT4 zero-points to GPU
        int4ZpBuf = D3D12Bindings.createDefaultBuffer(dev, zpBytes, arena);
        D3D12Bindings.uploadBytes(dev, queue, int4ZpBuf, zeroPoints, arena);

        // Create input and output GPU buffers (same size as legacy path)
        long inputBytes = (long) K * Float.BYTES;
        long outputBytes = (long) N * Float.BYTES;
        inputBuf = D3D12Bindings.createDefaultBuffer(dev, inputBytes, arena);
        outputBuf = D3D12Bindings.createDefaultBuffer(dev, outputBytes, arena);

        // Pre-allocate exec infrastructure (staging, cmd allocator, fence, barriers)
        // Pass null for dml — INT4 mode does not use DirectML
        prepareExecInfraInt4(dev, inputBytes, outputBytes);

        // Compile INT4 MatVec shader (on the shared execCmdList for MH caching).
        // WARP benefits from reducing multiple output rows in one thread group;
        // hardware GPUs keep the simpler one-thread-per-row shader unless explicitly
        // forced through the WARP backend.
        try {
            int4Shader = new GpuComputeKernel(wb, execCmdList,
                    useWarpParallelInt4Matvec ? warpParallelMatvecShaderSource(warpParallelInt4MatvecRows) : INT4_MATVEC_HLSL,
                    useWarpParallelInt4Matvec ? "int4_matvec_warp_parallel_rows" + warpParallelInt4MatvecRows : "int4_matvec",
                    5,   // 5 UAVs: X, QW, Scales, ZP, Y
                    3,   // 3 constants: N, K, blockSize
                    useWarpParallelInt4Matvec ? warpParallelInt4MatvecGroupSize(warpParallelInt4MatvecRows) : 64);
        } catch (RuntimeException ex) {
            if (!useWarpParallelInt4Matvec || warpParallelInt4MatvecRows != WARP_PARALLEL_INT4_MATVEC_ROWS32) {
                throw ex;
            }
            log.warn("WARP ROWS=32 INT4 matvec shader failed to compile for [{},{}]; falling back to ROWS=16. Reason: {}",
                    N, K, ex.toString());
            warpParallelInt4MatvecRows = WARP_PARALLEL_INT4_MATVEC_ROWS16;
            int4Shader = new GpuComputeKernel(wb, execCmdList,
                    INT4_MATVEC_WARP_PARALLEL_HLSL,
                    "int4_matvec_warp_parallel_rows16",
                    5,
                    3,
                    warpParallelInt4MatvecGroupSize(warpParallelInt4MatvecRows));
        }

        // Pre-cache GPU VAs and constants (fixed for lifetime of this kernel)
        int4UavAddrs = new long[]{
                D3D12Bindings.getGpuVirtualAddress(inputBuf),
                D3D12Bindings.getGpuVirtualAddress(int4WeightBuf),
                D3D12Bindings.getGpuVirtualAddress(int4ScalesBuf),
                D3D12Bindings.getGpuVirtualAddress(int4ZpBuf),
                D3D12Bindings.getGpuVirtualAddress(outputBuf)
        };
        int4Constants = new int[]{N, K, blockSize};

        prepared = true;
        log.info("MatMulNBitsKernel (INT4 GPU) ready: [{}, {}] — {}/{} bytes on GPU vs {} legacy, matvecShader={}",
                N, K, qBytes + scaleBytes + zpBytes,
                qBytes + scaleBytes + zpBytes,
                (long) N * K * 4,
                useWarpParallelInt4Matvec ? "warp-parallel-rows" + warpParallelInt4MatvecRows : "row-serial");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Preparation: upload FP32 weights → compile → pre-allocate exec infra
    // ══════════════════════════════════════════════════════════════════════

    /**
     * GPU setup: create buffers, upload weights, compile GEMM, pre-allocate
     * execution infrastructure.  Called by both constructors.
     */
    private void prepareGpu(float[] dequantized) throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        var queue = wb.getCommandQueue();
        var dml = wb.getDmlDevice();

        // ── Step 1: Create GPU buffers ────────────────────────────────
        // The weight matrix is [N, K] in row-major order.
        // Each weight = (nibble_value - zero_point) * scale
        long weightBytes = (long) N * K * Float.BYTES;
        long inputBytes = (long) K * Float.BYTES;        // M=1 for matvec
        long outputBytes = (long) N * Float.BYTES;
        long biasBytes = (long) N * Float.BYTES;

        weightBuf = D3D12Bindings.createDefaultBuffer(dev, weightBytes, arena);
        biasBuf = D3D12Bindings.createDefaultBuffer(dev, biasBytes, arena);
        inputBuf = D3D12Bindings.createDefaultBuffer(dev, inputBytes, arena);
        outputBuf = D3D12Bindings.createDefaultBuffer(dev, outputBytes, arena);

        // ── Step 2: Upload weight data to GPU ─────────────────────────
        D3D12Bindings.uploadFloats(dev, queue, weightBuf, dequantized, arena);
        float[] zeroBias = new float[N]; // GEMM C tensor = zero bias
        D3D12Bindings.uploadFloats(dev, queue, biasBuf, zeroBias, arena);
        log.info("Uploaded weight [{}, {}] to GPU ({} MB)",
                N, K, weightBytes / (1024 * 1024));

        // ── Step 3: Create and compile DirectML GEMM operator ─────────
        MemorySegment gemm = arena.allocate(56, 8);
        gemm.set(ValueLayout.ADDRESS, 0, td(new int[]{1, 1, 1, K}));       // A: [1,1,1,K]
        gemm.set(ValueLayout.ADDRESS, 8, td(new int[]{1, 1, N, K}));       // B: [1,1,N,K]
        gemm.set(ValueLayout.ADDRESS, 16, td(new int[]{1, 1, 1, N}));       // C: [1,1,1,N]
        gemm.set(ValueLayout.ADDRESS, 24, td(new int[]{1, 1, 1, N}));       // Y: [1,1,1,N]
        gemm.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);       // transA
        gemm.set(ValueLayout.JAVA_INT, 36, DirectMlBindings.DML_MATRIX_TRANSFORM_TRANSPOSE);  // transB
        gemm.set(ValueLayout.JAVA_FLOAT, 40, 1.0f);  // alpha
        gemm.set(ValueLayout.JAVA_FLOAT, 44, 1.0f);  // beta
        gemm.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL);  // no fused activation

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_GEMM, gemm);
        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        compiledGemm = DirectMlBindings.compileOperator(dml, op,
                DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);

        // ── Step 4: Query binding properties ──────────────────────────
        long[] props = DirectMlBindings.getBindingProperties(compiledGemm, arena);
        descCount = Math.max((int) props[0], 1);
        tempSize = props[1];
        persistSize = props[2];
        log.debug("GEMM binding: desc={}, temp={}, persist={}", descCount, tempSize, persistSize);

        // ── Step 5: Create descriptor heap ────────────────────────────
        // Need descriptors for: initialization + execution
        int totalDesc = descCount * 2 + 4;
        descriptorHeap = D3D12Bindings.createDescriptorHeap(dev, totalDesc, arena);
        descriptorIncrement = D3D12Bindings.getDescriptorIncrementSize(dev);
        cmdRecorder = DirectMlBindings.createCommandRecorder(dml, arena);

        // ── Step 6: Allocate temp/persist buffers ─────────────────────
        if (tempSize > 0) {
            tempBuf = D3D12Bindings.createDefaultBuffer(dev, tempSize, arena);
        }
        if (persistSize > 0) {
            persistBuf = D3D12Bindings.createDefaultBuffer(dev, persistSize, arena);
        }

        // ── Step 7: Initialize the operator ───────────────────────────
        initializeOperator(dev, queue, dml);

        // ── Step 8: Pre-allocate execution infrastructure (V1.1) ──────
        prepareExecInfra(dev, dml, inputBytes, outputBytes);

        prepared = true;
        log.info("MatMulNBitsKernel ready: [{}, {}] on GPU (optimized single-submit)", N, K);
    }

    private int matvecDispatchElementCount() {
        if (!useWarpParallelInt4Matvec) {
            return N;
        }

        int rowGroups = (N + warpParallelInt4MatvecRows - 1) / warpParallelInt4MatvecRows;
        return Math.multiplyExact(rowGroups, warpParallelInt4MatvecGroupSize(warpParallelInt4MatvecRows));
    }

    // ── INT4-only execution infrastructure (no DML, no descriptor heap) ─────

    /**
     * Like {@link #prepareExecInfra} but skips all DirectML-specific setup.
     * Used by the INT4 GPU path which drives a custom HLSL compute shader.
     */
    private void prepareExecInfraInt4(MemorySegment dev, long inputBytes, long outputBytes)
            throws WindowsNativeException {

        uploadBuf = D3D12Bindings.createUploadBuffer(dev, inputBytes, arena);
        readbackBuf = D3D12Bindings.createReadbackBuffer(dev, outputBytes, arena);
        mappedUpload = D3D12Bindings.mapResource(uploadBuf, arena);
        mappedReadback = D3D12Bindings.mapResource(readbackBuf, arena);

        execAllocator = D3D12Bindings.createCommandAllocator(dev,
                D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        execCmdList = D3D12Bindings.createCommandList(dev,
                D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, execAllocator, arena);
        D3D12Bindings.closeCommandList(execCmdList);

        execFence = D3D12Bindings.createFence(dev, 0, arena);
        fenceValue = 0;

        // Barrier: input COPY_DEST → UAV (used before INT4 dispatch)
        barrierInputToUAV = allocTransitionBarrier(inputBuf,
                D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        // Barrier: output UAV → COPY_SOURCE
        barrierOutputToCS = allocTransitionBarrier(outputBuf,
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);
        // Cleanup barriers
        barrierInputToCommon = allocTransitionBarrier(inputBuf,
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);
        barrierOutputToCommon = allocTransitionBarrier(outputBuf,
                D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);

        // heapArrayPtr not used in INT4 mode (no descriptor heap needed)
        // cmdListArrayPtr used by standalone matvec()
        cmdListArrayPtr = arena.allocate(ValueLayout.ADDRESS);
        cmdListArrayPtr.set(ValueLayout.ADDRESS, 0, execCmdList);

        // Pre-cache MethodHandles needed by both INT4 dispatch and standalone matvec()
        var queue = wb.getCommandQueue();
        mhResetAllocator = DxgiBindings.vtableMethod(execAllocator, 8,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhResetCmdList = DxgiBindings.vtableMethod(execCmdList, 10,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhCopyBufferRegion = DxgiBindings.vtableMethod(execCmdList, 15,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_LONG));
        mhResourceBarrier = DxgiBindings.vtableMethod(execCmdList, 26,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhCloseCmdList = DxgiBindings.vtableMethod(execCmdList, 9,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhExecuteCmdLists = DxgiBindings.vtableMethod(queue, 10,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhQueueSignal = DxgiBindings.vtableMethod(queue, 14,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        mhFenceGetCompleted = DxgiBindings.vtableMethod(execFence, 8,
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
        // mhRecordDispatch, mhSetDescriptorHeaps, heapArrayPtr left null — not used in INT4 mode
    }

    // ── Pre-cached MethodHandles for zero-overhead hot path (V1.2) ──────
    private MethodHandle mhResetAllocator;
    private MethodHandle mhResetCmdList;
    private MethodHandle mhCopyBufferRegion;
    private MethodHandle mhResourceBarrier;
    private MethodHandle mhSetDescriptorHeaps;
    private MethodHandle mhCloseCmdList;
    private MethodHandle mhExecuteCmdLists;
    private MethodHandle mhQueueSignal;
    private MethodHandle mhFenceGetCompleted;
    private MethodHandle mhRecordDispatch;

    /**
     * Pre-allocate all per-call resources: staging buffers (persistently mapped),
     * command allocator, command list, fence, binding table, barrier structs,
     * and MethodHandle cache.
     */
    private void prepareExecInfra(MemorySegment dev, MemorySegment dml,
                                  long inputBytes, long outputBytes)
            throws WindowsNativeException {

        // Staging buffers with persistent CPU mapping
        uploadBuf = D3D12Bindings.createUploadBuffer(dev, inputBytes, arena);
        readbackBuf = D3D12Bindings.createReadbackBuffer(dev, outputBytes, arena);
        mappedUpload = D3D12Bindings.mapResource(uploadBuf, arena);
        mappedReadback = D3D12Bindings.mapResource(readbackBuf, arena);

        // Reusable command allocator + command list
        execAllocator = D3D12Bindings.createCommandAllocator(dev,
                D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        execCmdList = D3D12Bindings.createCommandList(dev,
                D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, execAllocator, arena);
        D3D12Bindings.closeCommandList(execCmdList); // close so we can Reset it later

        // Reusable fence
        execFence = D3D12Bindings.createFence(dev, 0, arena);
        fenceValue = 0;

        // ── Pre-allocate barrier structs (V1.2 — eliminates per-call Arena) ──
        // Transition barrier: input COPY_DEST → UAV
        barrierInputToUAV = allocTransitionBarrier(inputBuf,
                D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        // V1.3: barrierUAV removed — the transition barrier output UAV→COPY_SOURCE
        // already provides necessary synchronization for DML UAV writes.
        // Transition barrier: output UAV → COPY_SOURCE
        barrierOutputToCS = allocTransitionBarrier(outputBuf,
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);
        // Transition barrier: input UAV → COMMON
        barrierInputToCommon = allocTransitionBarrier(inputBuf,
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);
        // Transition barrier: output COPY_SOURCE → COMMON
        barrierOutputToCommon = allocTransitionBarrier(outputBuf,
                D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);
        // Heap array for SetDescriptorHeaps
        heapArrayPtr = arena.allocate(ValueLayout.ADDRESS);
        heapArrayPtr.set(ValueLayout.ADDRESS, 0, descriptorHeap);
        // CmdList array for ExecuteCommandLists
        cmdListArrayPtr = arena.allocate(ValueLayout.ADDRESS);
        cmdListArrayPtr.set(ValueLayout.ADDRESS, 0, execCmdList);

        // ── Pre-cache MethodHandles for the hot path (V1.2) ──────────────
        var queue = wb.getCommandQueue();
        // ID3D12CommandAllocator::Reset (vtable slot 8)
        mhResetAllocator = DxgiBindings.vtableMethod(execAllocator, 8,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // ID3D12GraphicsCommandList::Reset (vtable slot 10) — (this, pAllocator, pInitialState)
        mhResetCmdList = DxgiBindings.vtableMethod(execCmdList, 10,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        // ID3D12GraphicsCommandList::CopyBufferRegion (vtable slot 15)
        mhCopyBufferRegion = DxgiBindings.vtableMethod(execCmdList, 15,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_LONG));
        // ID3D12GraphicsCommandList::ResourceBarrier (vtable slot 26)
        mhResourceBarrier = DxgiBindings.vtableMethod(execCmdList, 26,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // ID3D12GraphicsCommandList::SetDescriptorHeaps (vtable slot 28)
        mhSetDescriptorHeaps = DxgiBindings.vtableMethod(execCmdList, 28,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // ID3D12GraphicsCommandList::Close (vtable slot 9)
        mhCloseCmdList = DxgiBindings.vtableMethod(execCmdList, 9,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // ID3D12CommandQueue::ExecuteCommandLists (vtable slot 10)
        mhExecuteCmdLists = DxgiBindings.vtableMethod(queue, 10,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        // ID3D12CommandQueue::Signal (vtable slot 14)
        mhQueueSignal = DxgiBindings.vtableMethod(queue, 14,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        // ID3D12Fence::GetCompletedValue (vtable slot 8)
        mhFenceGetCompleted = DxgiBindings.vtableMethod(execFence, 8,
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
        // IDMLCommandRecorder::RecordDispatch (vtable slot RECORDER_RECORD_DISPATCH)
        mhRecordDispatch = DxgiBindings.vtableMethod(cmdRecorder,
                DirectMlBindings.RECORDER_RECORD_DISPATCH,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS));

        // Reusable execution binding table (bindings never change between calls)
        long cpuBase = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        long gpuBase = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        int descOff = descCount + 4;

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, compiledGemm,
                cpuBase + (long) descOff * descriptorIncrement,
                gpuBase + (long) descOff * descriptorIncrement, descCount);
        execBindingTable = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        // Bind inputs: A=input, B=weight, C=bias (static — only data in inputBuf changes)
        long weightBytes = (long) N * K * Float.BYTES;
        long biasBytes = (long) N * Float.BYTES;

        MemorySegment inputs = arena.allocate(16L * 3, 8);
        setBufferBinding(inputs, 0, inputBuf, inputBytes);
        setBufferBinding(inputs, 1, weightBuf, weightBytes);
        setBufferBinding(inputs, 2, biasBuf, biasBytes);
        DirectMlBindings.bindInputs(execBindingTable, 3, inputs);

        // Bind output
        MemorySegment outputs = arena.allocate(16, 8);
        setBufferBinding(outputs, 0, outputBuf, outputBytes);
        DirectMlBindings.bindOutputs(execBindingTable, 1, outputs);

        // Bind temp/persist
        if (tempSize > 0 && tempBuf != null) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, tempBuf, 0, tempSize);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena,
                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindTemporaryResource(execBindingTable, bd);
        }
        if (persistSize > 0 && persistBuf != null) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, persistBuf, 0, persistSize);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena,
                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindPersistentResource(execBindingTable, bd);
        }

        log.debug("Execution infrastructure pre-allocated (upload={}, readback={}, " +
                        "fence, cmdList, bindingTable, barriers, {} cached MethodHandles)",
                inputBytes, outputBytes, 10);
    }

    /**
     * Allocate a reusable transition barrier struct in the kernel's arena.
     */
    private MemorySegment allocTransitionBarrier(MemorySegment resource, int before, int after) {
        MemorySegment b = arena.allocate(32, 8);
        b.set(ValueLayout.JAVA_INT, 0, D3D12Bindings.D3D12_RESOURCE_BARRIER_TYPE_TRANSITION);
        b.set(ValueLayout.JAVA_INT, 4, D3D12Bindings.D3D12_RESOURCE_BARRIER_FLAG_NONE);
        b.set(ValueLayout.ADDRESS, 8, resource);
        b.set(ValueLayout.JAVA_INT, 16, D3D12Bindings.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
        b.set(ValueLayout.JAVA_INT, 20, before);
        b.set(ValueLayout.JAVA_INT, 24, after);
        return b;
    }

    private void initializeOperator(MemorySegment dev, MemorySegment queue,
                                    MemorySegment dml) throws WindowsNativeException {
        MemorySegment initializer = DirectMlBindings.createOperatorInitializer(
                dml, new MemorySegment[]{compiledGemm}, arena);

        long[] initProps = DirectMlBindings.getBindingProperties(initializer, arena);
        int initDescCount = Math.max((int) initProps[0], 1);
        long initTempSize = initProps[1];

        long cpuBase = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        long gpuBase = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);

        MemorySegment initBtDesc = DirectMlBindings.allocBindingTableDesc(arena, initializer,
                cpuBase, gpuBase, initDescCount);
        MemorySegment initBt = DirectMlBindings.createBindingTable(dml, initBtDesc, arena);

        // Bind persistent resource to initializer output
        if (persistSize > 0 && persistBuf != null) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, persistBuf, 0, persistSize);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena,
                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindOutputs(initBt, 1, bd);
        }

        // Bind temp resource for initialization
        MemorySegment initTempBuf = null;
        if (initTempSize > 0) {
            initTempBuf = D3D12Bindings.createDefaultBuffer(dev, initTempSize, arena);
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, initTempBuf, 0, initTempSize);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena,
                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindTemporaryResource(initBt, bd);
        }

        // Record and execute initialization
        var alloc = D3D12Bindings.createCommandAllocator(dev,
                D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdList = null;
        try {
            cmdList = D3D12Bindings.createCommandList(dev,
                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, arena);
            D3D12Bindings.setDescriptorHeaps(cmdList, descriptorHeap, arena);
            DirectMlBindings.recordDispatch(cmdRecorder, cmdList, initializer, initBt);
            D3D12Bindings.executeAndWait(dev, queue, cmdList, arena);
        } finally {
            if (cmdList != null) DxgiBindings.release(cmdList);
            DxgiBindings.release(alloc);
            DxgiBindings.release(initBt);
            DxgiBindings.release(initializer);
            if (initTempBuf != null) DxgiBindings.release(initTempBuf);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Inference: matvec on GPU (optimized single-submit)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Compute y = x @ W^T on GPU, returning a freshly allocated result array.
     *
     * @param x input vector [K]
     * @return output vector [N] (freshly allocated)
     */
    public float[] matvec(float[] x) {
        float[] result = new float[N];
        matvec(x, result);
        return result;
    }

    /**
     * Compute y = x @ W^T on GPU, writing result into a caller-provided buffer.
     * <p>
     * <b>V1.2 zero-alloc hot path</b>: upload, dispatch, and readback are combined
     * into a single command list submission with a single fence wait.
     * <p>
     * <b>V3.0 INT4 mode</b>: dispatches a custom HLSL compute shader reading
     * packed INT4 weights directly from VRAM — ~8× less memory bandwidth than
     * the legacy FP32/DML GEMM path.
     *
     * @param x   input vector [K]
     * @param out output vector [N] (must have length ≥ N)
     */
    public void matvec(float[] x, float[] out) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");
        if (x.length != K) throw new IllegalArgumentException(
                "Input length " + x.length + " != K=" + K);

        long inputBytes = (long) K * Float.BYTES;
        long outputBytes = (long) N * Float.BYTES;

        try {
            // 1. Write input to persistently-mapped upload buffer
            MemorySegment.copy(x, 0, mappedUpload, ValueLayout.JAVA_FLOAT, 0, K);

            // 2. Reset allocator + command list (pre-cached MethodHandles)
            int hr = (int) mhResetAllocator.invokeExact(execAllocator);
            HResult.check(hr, "CommandAllocator::Reset");
            hr = (int) mhResetCmdList.invokeExact(execCmdList, execAllocator, MemorySegment.NULL);
            HResult.check(hr, "CommandList::Reset");

            // 3. Record: upload → barrier → dispatch → barrier → readback
            mhCopyBufferRegion.invokeExact(execCmdList, inputBuf, 0L, uploadBuf, 0L, inputBytes);
            mhResourceBarrier.invokeExact(execCmdList, 1, barrierInputToUAV);
            if (useInt4Gpu) {
                // V3.0: custom HLSL INT4 compute dispatch — no descriptor heap needed
                int4Shader.recordDispatch(execCmdList, int4UavAddrs, int4Constants, matvecDispatchElementCount());
            } else {
                // Legacy: DML GEMM with FP32 weight buffer
                mhSetDescriptorHeaps.invokeExact(execCmdList, 1, heapArrayPtr);
                mhRecordDispatch.invokeExact(cmdRecorder, execCmdList, compiledGemm, execBindingTable);
            }
            mhResourceBarrier.invokeExact(execCmdList, 1, barrierOutputToCS);
            mhCopyBufferRegion.invokeExact(execCmdList, readbackBuf, 0L, outputBuf, 0L, outputBytes);
            mhResourceBarrier.invokeExact(execCmdList, 1, barrierInputToCommon);
            mhResourceBarrier.invokeExact(execCmdList, 1, barrierOutputToCommon);

            // 4. Close + Execute SINGLE combined command list + fence signal
            hr = (int) mhCloseCmdList.invokeExact(execCmdList);
            HResult.check(hr, "CommandList::Close");
            mhExecuteCmdLists.invokeExact(wb.getCommandQueue(), 1, cmdListArrayPtr);

            fenceValue++;
            hr = (int) mhQueueSignal.invokeExact(wb.getCommandQueue(), execFence, fenceValue);
            HResult.check(hr, "Queue::Signal");

            // 5. Spin-wait for GPU completion (pre-cached fence MethodHandle)
            long deadline = System.currentTimeMillis() + 120_000;
            while ((long) mhFenceGetCompleted.invokeExact(execFence) < fenceValue) {
                if (System.currentTimeMillis() > deadline) {
                    throw new WindowsNativeException(
                            "GPU fence timeout after 120000 ms – the GPU may be hung");
                }
                Thread.onSpinWait();
            }

            // 6. Read result from persistently-mapped readback buffer
            MemorySegment.copy(mappedReadback, ValueLayout.JAVA_FLOAT, 0, out, 0, N);

        } catch (WindowsNativeException e) {
            throw new RuntimeException("MatMulNBitsKernel.matvec failed", e);
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.matvec failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Pipeline-batched execution (V2.0 — record into external GpuPipeline)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Record this kernel's DML dispatch into an external {@link GpuPipeline}'s command list.
     * <p>
     * <b>V2.0 submission collapse</b>: Instead of a self-contained submit+wait cycle,
     * this method only records the upload, barriers, DML dispatch, and readback into
     * the pipeline's command list. No execution happens until
     * {@link GpuPipeline#submitAndWait()} is called.
     * <p>
     * This allows batching multiple kernel dispatches into ONE command list submission
     * with ONE fence wait, eliminating per-kernel synchronization overhead.
     *
     * @param pipeline the shared GPU pipeline (must be in recording state)
     * @param x        input vector [K]
     * @param out      output vector [N] (filled after pipeline.submitAndWait + pipeline.readbackInto)
     */
    public void recordInto(GpuPipeline pipeline, float[] x) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");
        if (x.length != K) throw new IllegalArgumentException(
                "Input length " + x.length + " != K=" + K);

        long inputBytes = (long) K * Float.BYTES;
        long outputBytes = (long) N * Float.BYTES;

        try {
            MemorySegment.copy(x, 0, mappedUpload, ValueLayout.JAVA_FLOAT, 0, K);
            var cl = pipeline.getCommandList();
            mhCopyBufferRegion.invokeExact(cl, inputBuf, 0L, uploadBuf, 0L, inputBytes);
            mhResourceBarrier.invokeExact(cl, 1, barrierInputToUAV);
            if (useInt4Gpu) {
                int4Shader.recordDispatch(cl, int4UavAddrs, int4Constants, matvecDispatchElementCount());
            } else {
                mhSetDescriptorHeaps.invokeExact(cl, 1, heapArrayPtr);
                mhRecordDispatch.invokeExact(cmdRecorder, cl, compiledGemm, execBindingTable);
            }
            mhResourceBarrier.invokeExact(cl, 1, barrierOutputToCS);
            mhCopyBufferRegion.invokeExact(cl, readbackBuf, 0L, outputBuf, 0L, outputBytes);
            mhResourceBarrier.invokeExact(cl, 1, barrierInputToCommon);
            mhResourceBarrier.invokeExact(cl, 1, barrierOutputToCommon);
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.recordInto failed", t);
        }
    }

    /**
     * Read the result from this kernel's readback buffer into a Java array.
     * Must be called AFTER the pipeline has been submitted and waited.
     *
     * @param out destination array [N]
     */
    public void readResult(float[] out) {
        MemorySegment.copy(mappedReadback, ValueLayout.JAVA_FLOAT, 0, out, 0, N);
    }

    /**
     * Record this kernel's dispatch that reads input from a GPU-resident buffer.
     *
     * @param pipeline      the shared GPU pipeline (recording state)
     * @param gpuInputBuf   GPU default buffer containing the input [K] floats
     * @param gpuInputBytes byte size of the input data
     */
    public void recordIntoGpuResident(GpuPipeline pipeline, MemorySegment gpuInputBuf,
                                      long gpuInputBytes) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");

        long outputBytes = (long) N * Float.BYTES;

        try {
            var cl = pipeline.getCommandList();
            mhCopyBufferRegion.invokeExact(cl, inputBuf, 0L, gpuInputBuf, 0L, gpuInputBytes);
            mhResourceBarrier.invokeExact(cl, 1, barrierInputToUAV);
            if (useInt4Gpu) {
                int4Shader.recordDispatch(cl, int4UavAddrs, int4Constants, matvecDispatchElementCount());
            } else {
                mhSetDescriptorHeaps.invokeExact(cl, 1, heapArrayPtr);
                mhRecordDispatch.invokeExact(cmdRecorder, cl, compiledGemm, execBindingTable);
            }
            mhResourceBarrier.invokeExact(cl, 1, barrierOutputToCS);
            // Leave output in COPY_SOURCE state (caller chains further or reads back)
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.recordIntoGpuResident failed", t);
        }
    }

    /**
     * Record the cleanup barriers to return buffers to COMMON state.
     * Call after the last operation in a batch that uses this kernel's output.
     */
    public void recordCleanupBarriers(GpuPipeline pipeline) {
        try {
            var cl = pipeline.getCommandList();
            mhResourceBarrier.invokeExact(cl, 1, barrierInputToCommon);
            mhResourceBarrier.invokeExact(cl, 1, barrierOutputToCommon);
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.recordCleanupBarriers failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // MLP-batch mode (V2.0 — no readback, no cleanup, output stays in UAV)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Record upload from CPU + DML dispatch for batch mode.
     * <p>
     * Output buffer stays in UAV state after the dispatch — caller is responsible
     * for any subsequent transitions and barriers. No readback copy is recorded.
     * <p>
     * After the batch's {@link GpuPipeline#submitAndWait()}, all buffers automatically
     * decay back to COMMON state (D3D12 buffer decay rule).
     *
     * @param pipeline shared GPU pipeline (recording state)
     * @param x        input vector [K] from CPU
     */
    public void recordBatchFromCpu(GpuPipeline pipeline, float[] x) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");
        if (x.length != K) throw new IllegalArgumentException(
                "Input length " + x.length + " != K=" + K);
        long inputBytes = (long) K * Float.BYTES;
        try {
            MemorySegment.copy(x, 0, mappedUpload, ValueLayout.JAVA_FLOAT, 0, K);
            var cl = pipeline.getCommandList();
            mhCopyBufferRegion.invokeExact(cl, inputBuf, 0L, uploadBuf, 0L, inputBytes);
            mhResourceBarrier.invokeExact(cl, 1, barrierInputToUAV);
            if (useInt4Gpu) {
                int4Shader.recordDispatch(cl, int4UavAddrs, int4Constants, matvecDispatchElementCount());
            } else {
                mhSetDescriptorHeaps.invokeExact(cl, 1, heapArrayPtr);
                mhRecordDispatch.invokeExact(cmdRecorder, cl, compiledGemm, execBindingTable);
            }
            // Output stays in UAV — no readback, no cleanup
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.recordBatchFromCpu failed", t);
        }
    }

    /**
     * Record dispatch from a GPU-resident input buffer.
     * Like {@link #recordBatchFromCpu} but the input is already on GPU.
     * Copies from {@code gpuSrcBuf} → inputBuf, barriers, dispatches,
     * leaves output in UAV. Used by Opt-C chained decode to route
     * the normed hidden state from one layer into the next layer's QKV.
     *
     * @param pipeline  shared GPU pipeline (recording state)
     * @param gpuSrcBuf GPU default buffer containing the input [K] floats
     */
    public void recordBatchFromGpu(GpuPipeline pipeline, MemorySegment gpuSrcBuf) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");
        long inputBytes = (long) K * Float.BYTES;
        try {
            var cl = pipeline.getCommandList();
            mhCopyBufferRegion.invokeExact(cl, inputBuf, 0L, gpuSrcBuf, 0L, inputBytes);
            mhResourceBarrier.invokeExact(cl, 1, barrierInputToUAV);
            if (useInt4Gpu) {
                int4Shader.recordDispatch(cl, int4UavAddrs, int4Constants, matvecDispatchElementCount());
            } else {
                mhSetDescriptorHeaps.invokeExact(cl, 1, heapArrayPtr);
                mhRecordDispatch.invokeExact(cmdRecorder, cl, compiledGemm, execBindingTable);
            }
            // Output stays in UAV
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.recordBatchFromGpu failed", t);
        }
    }

    /**
     * Record only the dispatch — input buffer already contains data in UAV state
     * (written by a preceding compute shader, e.g., RMSNorm or SwiGLU).
     * <p>
     * Caller must ensure a UAV barrier between the compute write and this dispatch.
     * Output buffer stays in UAV after dispatch.
     *
     * @param pipeline shared GPU pipeline (recording state)
     */
    public void recordBatchDispatchOnly(GpuPipeline pipeline) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");
        try {
            var cl = pipeline.getCommandList();
            if (useInt4Gpu) {
                int4Shader.recordDispatch(cl, int4UavAddrs, int4Constants, matvecDispatchElementCount());
            } else {
                mhSetDescriptorHeaps.invokeExact(cl, 1, heapArrayPtr);
                mhRecordDispatch.invokeExact(cmdRecorder, cl, compiledGemm, execBindingTable);
            }
            // Output stays in UAV
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.recordBatchDispatchOnly failed", t);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Batched matmul (Opt-A — prefill submission reduction)
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Whether this kernel supports the batched {@link #matmulBatch} path.
     * Currently only the INT4-GPU path supports batching; legacy DML/FP32 falls
     * back to per-row dispatches (callers should still use {@link #matvec}).
     */
    public boolean supportsBatch() {
        return prepared && batchEnabled;   // false if batch scratch alloc failed on this kernel
    }

    /**
     * Compute {@code Y = X @ W^T} for a batch of {@code M} input rows in ONE GPU
     * submission. Mathematically equivalent to {@code M} consecutive
     * {@link #matvec} calls but with a single fence-wait instead of {@code M}.
     *
     * <p>Batched scratch buffers are allocated lazily on first call and grown
     * (re-allocated) when {@code M} exceeds the previously-seen maximum.
     * Capacity is capped at {@link #MAX_BATCH_M} to keep VRAM bounded.
     *
     * @param xBatch   input rows packed row-major, length {@code >= M * K}
     * @param outBatch output rows packed row-major, length {@code >= M * N}
     * @param M        number of rows in this batch; values above {@link #MAX_BATCH_M} are split into chunks
     */
    public void matmulBatch(float[] xBatch, float[] outBatch, int M) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");
        if (M < 1) throw new IllegalArgumentException(
                "M out of range [1, " + MAX_BATCH_M + "]: " + M);
        if (xBatch.length < (long) M * K) throw new IllegalArgumentException(
                "xBatch too short: " + xBatch.length + " < " + ((long) M * K));
        if (outBatch.length < (long) M * N) throw new IllegalArgumentException(
                "outBatch too short: " + outBatch.length + " < " + ((long) M * N));
        if (M > MAX_BATCH_M) {
            matmulBatchChunked(xBatch, outBatch, M);
            return;
        }

        // If batch scratch allocation previously failed, fall back to per-row matvec.
        // This avoids GPU OOM on tight-VRAM Intel iGPUs while keeping the INT4 path
        // for decode (single-row matvec) and for layers where the batch alloc succeeded.
        if (!batchEnabled) {
            float[] row = new float[K];
            float[] rowOut = new float[N];
            for (int i = 0; i < M; i++) {
                System.arraycopy(xBatch, i * K, row, 0, K);
                matvec(row, rowOut);
                System.arraycopy(rowOut, 0, outBatch, i * N, N);
            }
            return;
        }

        try {
            ensureBatchCapacity(M);
        } catch (Throwable e) {
            log.warn("matmulBatch: failed to alloc batch scratch for [{},{}] M={} — disabling batching, falling back to per-row matvec ({})",
                    N, K, M, e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
            batchEnabled = false;
            float[] row = new float[K];
            float[] rowOut = new float[N];
            for (int i = 0; i < M; i++) {
                System.arraycopy(xBatch, i * K, row, 0, K);
                matvec(row, rowOut);
                System.arraycopy(rowOut, 0, outBatch, i * N, N);
            }
            return;
        }

        long inputBytes = (long) M * K * Float.BYTES;
        long outputBytes = (long) M * N * Float.BYTES;
        // INT4 mode: {N, K, blockSize, M, nTiles}; FP32 mode (tiled): {N, K, M, nTiles}
        int nTilesFp32 = (N + 15) / 16;
        int mTilesFp32 = (M + 15) / 16;
        int nTiles = useInt4Gpu ? (N + 15) / 16 : nTilesFp32;
        int[] batchConstants = useInt4Gpu
                ? new int[]{N, K, int4BlockSize, M, nTiles}
                : new int[]{N, K, M, nTilesFp32};
        long[] batchUavs = useInt4Gpu
                ? new long[]{
                D3D12Bindings.getGpuVirtualAddress(int4BatchInputBuf),
                D3D12Bindings.getGpuVirtualAddress(int4WeightBuf),
                D3D12Bindings.getGpuVirtualAddress(int4ScalesBuf),
                D3D12Bindings.getGpuVirtualAddress(int4ZpBuf),
                D3D12Bindings.getGpuVirtualAddress(int4BatchOutputBuf)}
                : new long[]{
                D3D12Bindings.getGpuVirtualAddress(int4BatchInputBuf),
                D3D12Bindings.getGpuVirtualAddress(weightBuf),
                D3D12Bindings.getGpuVirtualAddress(int4BatchOutputBuf)};
        GpuComputeKernel batchShader = useInt4Gpu ? sharedInt4BatchShader : sharedFp32BatchShader;
        // Tiled mode (both INT4 and FP32): 1 thread group per [16,16] output tile
        // (groupSize=256). elementCount = mTiles*nTiles*256 threads total.
        int dispatchElements = mTilesFp32 * nTiles * 256;

        try {
            // 1. Upload xBatch to persistently-mapped upload buffer.
            MemorySegment.copy(xBatch, 0, int4BatchMappedUpload,
                    ValueLayout.JAVA_FLOAT, 0, M * K);

            // 2. Reset allocator + command list.
            int hr = (int) mhResetAllocator.invokeExact(execAllocator);
            HResult.check(hr, "CommandAllocator::Reset");
            hr = (int) mhResetCmdList.invokeExact(execCmdList, execAllocator, MemorySegment.NULL);
            HResult.check(hr, "CommandList::Reset");

            // 3. Record: upload → barrier → dispatch → barrier → readback.
            mhCopyBufferRegion.invokeExact(execCmdList,
                    int4BatchInputBuf, 0L, int4BatchUploadBuf, 0L, inputBytes);
            mhResourceBarrier.invokeExact(execCmdList, 1, barrierBatchInputToUAV);
            batchShader.recordDispatch(execCmdList, batchUavs, batchConstants, dispatchElements);
            mhResourceBarrier.invokeExact(execCmdList, 1, barrierBatchOutputToCS);
            mhCopyBufferRegion.invokeExact(execCmdList,
                    int4BatchReadbackBuf, 0L, int4BatchOutputBuf, 0L, outputBytes);
            mhResourceBarrier.invokeExact(execCmdList, 1, barrierBatchInputToCommon);
            mhResourceBarrier.invokeExact(execCmdList, 1, barrierBatchOutputToCommon);

            // 4. Close + execute + fence signal.
            hr = (int) mhCloseCmdList.invokeExact(execCmdList);
            HResult.check(hr, "CommandList::Close");
            mhExecuteCmdLists.invokeExact(wb.getCommandQueue(), 1, cmdListArrayPtr);

            fenceValue++;
            hr = (int) mhQueueSignal.invokeExact(wb.getCommandQueue(), execFence, fenceValue);
            HResult.check(hr, "Queue::Signal");

            // 5. Spin-wait for GPU completion.
            long deadline = System.currentTimeMillis() + 120_000;
            while ((long) mhFenceGetCompleted.invokeExact(execFence) < fenceValue) {
                if (System.currentTimeMillis() > deadline) {
                    throw new WindowsNativeException(
                            "GPU fence timeout after 120000 ms in matmulBatch(M=" + M + ")");
                }
                Thread.onSpinWait();
            }

            // 6. Read result.
            MemorySegment.copy(int4BatchMappedReadback, ValueLayout.JAVA_FLOAT, 0,
                    outBatch, 0, M * N);

        } catch (WindowsNativeException e) {
            throw new RuntimeException("MatMulNBitsKernel.matmulBatch failed", e);
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.matmulBatch failed", t);
        }
    }

    /**
     * Compute a long prefill batch by splitting it into bounded GPU batches.
     * <p>
     * The shared batch scratch buffers are intentionally capped at
     * {@link #MAX_BATCH_M} to avoid GPU memory spikes on WARP/iGPU systems.
     * Long prompts can still be supported by running several bounded batches
     * back-to-back. This keeps the existing memory cap while removing the
     * artificial 256-token prompt limit from the Workbench summarizer.
     */
    private void matmulBatchChunked(float[] xBatch, float[] outBatch, int M) {
        log.info("matmulBatch: splitting long batch M={} into chunks of <={} for [{},{}]",
                M, MAX_BATCH_M, N, K);
        float[] chunkIn = new float[MAX_BATCH_M * K];
        float[] chunkOut = new float[MAX_BATCH_M * N];
        int offsetRows = 0;
        while (offsetRows < M) {
            int rows = Math.min(MAX_BATCH_M, M - offsetRows);
            System.arraycopy(xBatch, offsetRows * K, chunkIn, 0, rows * K);
            matmulBatch(chunkIn, chunkOut, rows);
            System.arraycopy(chunkOut, 0, outBatch, offsetRows * N, rows * N);
            offsetRows += rows;
        }
    }

    /**
     * Allocate or grow the batched scratch buffers + barriers to hold at least
     * {@code requiredM} rows. Idempotent: calls with {@code requiredM <= batchCapacityM}
     * return immediately.
     *
     * <p>SHARED BUFFER OPTIMISATION (Opt-A-3, 2026-06-03):
     * The scratch buffers are now {@code static} — one set for ALL kernel instances.
     * During prefill, layers run sequentially (one matmulBatch() at a time), so
     * a single set of buffers can be safely reused. This drops VRAM from
     * 96×20MB → 1×20MB, fixing Intel iGPU crashes.
     *
     * <p>On first call also compiles the batched HLSL shader.
     */
    private void ensureBatchCapacity(int requiredM) throws WindowsNativeException {
        if (requiredM > MAX_BATCH_M) {
            throw new IllegalArgumentException(
                    "requiredM=" + requiredM + " exceeds MAX_BATCH_M=" + MAX_BATCH_M);
        }

        // ── Device-ownership guard ────────────────────────────────────────
        // If the static batch resources were built on a DIFFERENT (now-closed)
        // device — e.g. a previous model in the workbench — reusing them would
        // dispatch device-A buffers on device-B's command queue and remove the
        // device. Release the stale resources so they get rebuilt below for THIS
        // device.
        long currentDeviceAddr = wb.getD3d12Device().address();
        if (sharedBatchDeviceAddr != 0L && sharedBatchDeviceAddr != currentDeviceAddr) {
            synchronized (batchBufferLock) {
                if (sharedBatchDeviceAddr != 0L && sharedBatchDeviceAddr != currentDeviceAddr) {
                    log.info("Batch resources owned by a different D3D12 device (0x{} -> 0x{}); "
                                    + "releasing stale shared buffers/shaders and rebuilding for the new device",
                            Long.toHexString(sharedBatchDeviceAddr), Long.toHexString(currentDeviceAddr));
                    releaseStaticBatchResources();
                }
            }
        }

        // ── If shared buffers already exist, just wire instance fields ──
        if (requiredM <= sharedBatchCapacityM && K <= sharedBatchMaxK && N <= sharedBatchMaxN) {
            synchronized (batchBufferLock) {
                if (sharedInt4BatchInputBuf != null) {
                    this.int4BatchInputBuf = sharedInt4BatchInputBuf;
                    this.int4BatchOutputBuf = sharedInt4BatchOutputBuf;
                    this.int4BatchUploadBuf = sharedInt4BatchUploadBuf;
                    this.int4BatchReadbackBuf = sharedInt4BatchReadbackBuf;
                    this.int4BatchMappedUpload = sharedInt4BatchMappedUpload;
                    this.int4BatchMappedReadback = sharedInt4BatchMappedReadback;
                    this.batchCapacityM = sharedBatchCapacityM;
                    // Barriers for this instance (pointing to shared buffers)
                    barrierBatchInputToUAV = allocTransitionBarrier(sharedInt4BatchInputBuf,
                            D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    barrierBatchOutputToCS = allocTransitionBarrier(sharedInt4BatchOutputBuf,
                            D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);
                    barrierBatchInputToCommon = allocTransitionBarrier(sharedInt4BatchInputBuf,
                            D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);
                    barrierBatchOutputToCommon = allocTransitionBarrier(sharedInt4BatchOutputBuf,
                            D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE,
                            D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);
                    return;
                }
            }
        }

        var dev = wb.getD3d12Device();

        // ── STATIC BUFFER ALLOCATION (synchronised) ─────────────────────
        synchronized (batchBufferLock) {
            // Double-check under lock (another thread may have grown buffers)
            if (requiredM <= sharedBatchCapacityM && K <= sharedBatchMaxK && N <= sharedBatchMaxN && sharedInt4BatchInputBuf != null) {
                this.int4BatchInputBuf = sharedInt4BatchInputBuf;
                this.int4BatchOutputBuf = sharedInt4BatchOutputBuf;
                this.int4BatchUploadBuf = sharedInt4BatchUploadBuf;
                this.int4BatchReadbackBuf = sharedInt4BatchReadbackBuf;
                this.int4BatchMappedUpload = sharedInt4BatchMappedUpload;
                this.int4BatchMappedReadback = sharedInt4BatchMappedReadback;
                this.batchCapacityM = sharedBatchCapacityM;
                // Barriers for this instance (pointing to shared buffers)
                barrierBatchInputToUAV = allocTransitionBarrier(sharedInt4BatchInputBuf,
                        D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                barrierBatchOutputToCS = allocTransitionBarrier(sharedInt4BatchOutputBuf,
                        D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);
                barrierBatchInputToCommon = allocTransitionBarrier(sharedInt4BatchInputBuf,
                        D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);
                barrierBatchOutputToCommon = allocTransitionBarrier(sharedInt4BatchOutputBuf,
                        D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);
                return;
            }

            // Release previous (smaller) buffers if any
            if (sharedInt4BatchInputBuf != null) {
                D3D12Bindings.unmapResource(sharedInt4BatchUploadBuf);
                D3D12Bindings.unmapResource(sharedInt4BatchReadbackBuf);
                DxgiBindings.release(sharedInt4BatchInputBuf);
                DxgiBindings.release(sharedInt4BatchOutputBuf);
                DxgiBindings.release(sharedInt4BatchUploadBuf);
                DxgiBindings.release(sharedInt4BatchReadbackBuf);
            }

            int allocK = Math.max(K, sharedBatchMaxK);
            int allocN = Math.max(N, sharedBatchMaxN);
            long inputBytes = (long) requiredM * allocK * Float.BYTES;
            long outputBytes = (long) requiredM * allocN * Float.BYTES;

            // ── STATIC ARENA: allocate ONCE, reuse for all kernel instances ─────
            if (sharedBatchArena == null) {
                synchronized (sharedArenaLock) {
                    if (sharedBatchArena == null) {
                        sharedBatchArena = Arena.ofShared();
                        // Register shutdown hook to clean up static resources on JVM exit.
                        // Registered only ONCE for the JVM — the arena may be re-created
                        // when the owning device changes, but a single hook suffices.
                        if (!batchShutdownHookRegistered) {
                            batchShutdownHookRegistered = true;
                            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                                log.info("Shutdown hook: releasing static batch resources");
                                releaseStaticBatchResources();
                            }));
                        }
                    }
                }
            }
            // Record which device now owns the (re)built static batch resources so a
            // later model on a different device triggers a rebuild instead of reusing
            // these buffers cross-device.
            sharedBatchDeviceAddr = dev.address();

            // Allocate SHARED scratch buffers (static — one set for all instances)
            // NOTE: Uses sharedBatchArena, NOT instance 'arena'!
            sharedInt4BatchInputBuf = D3D12Bindings.createDefaultBuffer(dev, inputBytes, sharedBatchArena);
            sharedInt4BatchOutputBuf = D3D12Bindings.createDefaultBuffer(dev, outputBytes, sharedBatchArena);
            sharedInt4BatchUploadBuf = D3D12Bindings.createUploadBuffer(dev, inputBytes, sharedBatchArena);
            sharedInt4BatchReadbackBuf = D3D12Bindings.createReadbackBuffer(dev, outputBytes, sharedBatchArena);
            sharedInt4BatchMappedUpload = D3D12Bindings.mapResource(sharedInt4BatchUploadBuf, sharedBatchArena);
            sharedInt4BatchMappedReadback = D3D12Bindings.mapResource(sharedInt4BatchReadbackBuf, sharedBatchArena);

            // ── SHARED shader compilation (mode-dependent: INT4 vs FP32) ───
            if (useInt4Gpu) {
                if (sharedInt4BatchShader == null) {
                    synchronized (batchShaderLock) {
                        if (sharedInt4BatchShader == null) {
                            // Note: uses a TEMPORARY command list for compilation
                            // (shared shader outlives any single kernel instance)
                            var tempAlloc = D3D12Bindings.createCommandAllocator(dev,
                                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
                            var tempCL = D3D12Bindings.createCommandList(dev,
                                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, tempAlloc, arena);
                            D3D12Bindings.closeCommandList(tempCL);

                            sharedInt4BatchShader = new GpuComputeKernel(wb, tempCL,
                                    INT4_MATMUL_BATCH_TILED_HLSL, "int4_matmul_batch_weight_tiled",
                                    5,   // UAVs: X, QW, Scales, ZP, Y
                                    5,   // constants: N, K, blockSize, M, nTiles
                                    256);

                            DxgiBindings.release(tempCL);
                            DxgiBindings.release(tempAlloc);
                        }
                    }
                }
                sharedBatchMaxK = Math.max(sharedBatchMaxK, K);
                sharedBatchMaxN = Math.max(sharedBatchMaxN, N);
            } else {
                // FP32 / pre-dequantized weight path (used by qkvFused, gateUpFused).
                if (sharedFp32BatchShader == null) {
                    synchronized (batchShaderLock) {
                        if (sharedFp32BatchShader == null) {
                            var tempAlloc = D3D12Bindings.createCommandAllocator(dev,
                                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
                            var tempCL = D3D12Bindings.createCommandList(dev,
                                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, tempAlloc, arena);
                            D3D12Bindings.closeCommandList(tempCL);

                            sharedFp32BatchShader = new GpuComputeKernel(wb, tempCL,
                                    FP32_MATMUL_BATCH_HLSL, "fp32_matmul_batch_tiled",
                                    3,    // UAVs: X, W, Y
                                    4,    // constants: N, K, M, nTiles
                                    256); // 16×16 threads per output tile

                            DxgiBindings.release(tempCL);
                            DxgiBindings.release(tempAlloc);
                        }
                    }
                }
                sharedBatchMaxK = Math.max(sharedBatchMaxK, K);
                sharedBatchMaxN = Math.max(sharedBatchMaxN, N);
            }

            // ── Per-instance barrier structs (bind to shared buffers) ─────────
            // Barriers are instance-specific because they reference the SHARED buffers.
            barrierBatchInputToUAV = allocTransitionBarrier(sharedInt4BatchInputBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            barrierBatchOutputToCS = allocTransitionBarrier(sharedInt4BatchOutputBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);
            barrierBatchInputToCommon = allocTransitionBarrier(sharedInt4BatchInputBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);
            barrierBatchOutputToCommon = allocTransitionBarrier(sharedInt4BatchOutputBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);

            // ── Update INSTANCE references to point to SHARED buffers ─────────
            this.int4BatchInputBuf = sharedInt4BatchInputBuf;
            this.int4BatchOutputBuf = sharedInt4BatchOutputBuf;
            this.int4BatchUploadBuf = sharedInt4BatchUploadBuf;
            this.int4BatchReadbackBuf = sharedInt4BatchReadbackBuf;
            this.int4BatchMappedUpload = sharedInt4BatchMappedUpload;
            this.int4BatchMappedReadback = sharedInt4BatchMappedReadback;
            this.batchCapacityM = sharedBatchCapacityM;

            sharedBatchCapacityM = requiredM;
            long totalKb = (inputBytes + outputBytes) * 2 / 1024;
            log.info("MatMulNBitsKernel[N={}, K={}]: SHARED batch capacity grown to M={} ({} KB scratch, static)",
                    N, K, requiredM, totalKb);
        }
    }

    // ── GPU buffer accessors (for pipeline-batched operations) ─────────

    /**
     * GPU output buffer (default heap, UAV). Result is written here by DML dispatch.
     */
    public MemorySegment getOutputBuf() {
        return outputBuf;
    }

    /**
     * GPU input buffer (default heap, UAV).
     */
    public MemorySegment getInputBuf() {
        return inputBuf;
    }

    /**
     * Readback buffer (readback heap, mapped).
     */
    public MemorySegment getReadbackBuf() {
        return readbackBuf;
    }

    /**
     * Mapped readback pointer.
     */
    public MemorySegment getMappedReadback() {
        return mappedReadback;
    }

    /**
     * Mapped upload pointer.
     */
    public MemorySegment getMappedUpload() {
        return mappedUpload;
    }

    /**
     * Upload buffer.
     */
    public MemorySegment getUploadBuf() {
        return uploadBuf;
    }

    // ══════════════════════════════════════════════════════════════════════
    // INT4 dequantization (CPU, one-time)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Dequantize INT4 AWQ block-128 packed weights to FP32.
     * <p>
     * Each byte in {@code qWeight} contains 2 uint4 values (low nibble first).
     * Weight value = (nibble - zero_point) * scale
     * <p>
     * This method is public so callers can fuse multiple quantized weight
     * matrices (e.g., Q+K+V) by dequantizing separately, concatenating,
     * and creating a kernel via {@link #fromDequantizedWeights}.
     *
     * @return float[N * K] row-major weight matrix
     */
    public static float[] dequantizeInt4(byte[] qWeight, float[] scales, byte[] zeroPoints,
                                         int N, int K, int blockSize) {
        float[] result = new float[N * K];
        int blocksPerRow = K / blockSize;

        for (int n = 0; n < N; n++) {
            int qOffset = n * blocksPerRow * (blockSize / 2);
            int scaleOffset = n * blocksPerRow;

            for (int blk = 0; blk < blocksPerRow; blk++) {
                float scale = scales[scaleOffset + blk];

                // Zero point: 2 per byte, low nibble first
                int zpIdx = n * blocksPerRow + blk;
                int zpByte = zeroPoints[zpIdx / 2] & 0xFF;
                int zp = (zpIdx % 2 == 0) ? (zpByte & 0xF) : (zpByte >>> 4);

                int kBase = blk * blockSize;
                int qBase = qOffset + blk * (blockSize / 2);
                int rowBase = n * K;

                for (int j = 0; j < blockSize / 2; j++) {
                    int packed = qWeight[qBase + j] & 0xFF;
                    int w0 = (packed & 0xF) - zp;
                    int w1 = (packed >>> 4) - zp;
                    result[rowBase + kBase + 2 * j] = w0 * scale;
                    result[rowBase + kBase + 2 * j + 1] = w1 * scale;
                }
            }
        }
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Helpers
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Build a DML_TENSOR_DESC for FP32 buffer tensor.
     */
    private MemorySegment td(int[] sizes) {
        int elems = 1;
        for (int s : sizes) elems *= s;
        long byteSize = (long) elems * Float.BYTES;
        var bufTD = DirectMlBindings.allocBufferTensorDesc(arena,
                DirectMlBindings.DML_TENSOR_DATA_TYPE_FLOAT32, sizes, null, byteSize);
        return DirectMlBindings.allocTensorDesc(arena, bufTD);
    }

    /**
     * Set a DML_BINDING_DESC (buffer type) into an array at the given index.
     * Each binding desc is 16 bytes: Type(4)+pad(4)+Desc*(8).
     */
    private void setBufferBinding(MemorySegment array, int index,
                                  MemorySegment buffer, long sizeBytes) {
        long off = (long) index * 16;
        MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, buffer, 0, sizeBytes);
        array.set(ValueLayout.JAVA_INT, off, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
        array.set(ValueLayout.ADDRESS, off + 8, bb);
    }

    // ══════════════════════════════════════════════════════════════════════
    // AutoCloseable
    // ══════════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Unmap persistently-mapped staging buffers
        if (uploadBuf != null) D3D12Bindings.unmapResource(uploadBuf);
        if (readbackBuf != null) D3D12Bindings.unmapResource(readbackBuf);

        // Release INT4 GPU mode resources (if active)
        if (int4Shader != null) int4Shader.close();

        // ── STATIC RESOURCES: intentionally NOT released here ─────────────
        // SHARED shaders + SHARED batch scratch buffers live for the JVM lifetime.
        // Subsequent model loads reuse them without paying the ~30 s DXC cost.
        // (Opt-A-2 v2 fix, 2026-05-29; Opt-A-3 shared buffers, 2026-06-03).
        //
        // NOTE: We do NOT close sharedInt4BatchShader, sharedFp32BatchShader,
        //       sharedInt4BatchInputBuf, sharedInt4BatchOutputBuf, etc. here.
        //       They are released only when the JVM exits (or via a static shutdown hook).
        
        if (int4WeightBuf != null) DxgiBindings.release(int4WeightBuf);
        if (int4ScalesBuf != null) DxgiBindings.release(int4ScalesBuf);
        if (int4ZpBuf != null) DxgiBindings.release(int4ZpBuf);

        // ── Per-instance fields pointing to shared buffers: NULL them only ─────
        // (The actual GPU resources are static — NOT released here.)
        this.int4BatchInputBuf = null;
        this.int4BatchOutputBuf = null;
        this.int4BatchUploadBuf = null;
        this.int4BatchReadbackBuf = null;
        this.int4BatchMappedUpload = null;
        this.int4BatchMappedReadback = null;
        // int4BatchUavAddrs/fp32BatchUavAddrs: built dynamically in matmulBatch, no static refs to null

        // Release pre-allocated execution infrastructure (reverse creation order)
        if (execBindingTable != null) DxgiBindings.release(execBindingTable);
        if (execFence != null) DxgiBindings.release(execFence);
        if (execCmdList != null) DxgiBindings.release(execCmdList);
        if (execAllocator != null) DxgiBindings.release(execAllocator);
        if (readbackBuf != null) DxgiBindings.release(readbackBuf);
        if (uploadBuf != null) DxgiBindings.release(uploadBuf);

        // Release DML/FP32 operator resources (only in legacy mode)
        if (cmdRecorder != null) DxgiBindings.release(cmdRecorder);
        if (descriptorHeap != null) DxgiBindings.release(descriptorHeap);
        if (compiledGemm != null) DxgiBindings.release(compiledGemm);
        if (persistBuf != null) DxgiBindings.release(persistBuf);
        if (tempBuf != null) DxgiBindings.release(tempBuf);
        if (outputBuf != null) DxgiBindings.release(outputBuf);
        if (inputBuf != null) DxgiBindings.release(inputBuf);
        if (biasBuf != null) DxgiBindings.release(biasBuf);
        if (weightBuf != null) DxgiBindings.release(weightBuf);

        arena.close();
        log.trace("MatMulNBitsKernel closed [{}, {}] — shared buffers retained for JVM lifetime", N, K);
    }

    /**
     * Output features (rows of weight matrix).
     */
    public int getN() {
        return N;
    }

    /**
     * Input features (cols of weight matrix).
     */
    public int getK() {
        return K;
    }

    // ════════════════════════════════════════════════════════════════════
    // Static batch resource cleanup (JVM shutdown hook)
    // ════════════════════════════════════════════════════════════════════

    /**
     * Release ALL static batch resources (shared buffers + shared shaders).
     * Called by JVM shutdown hook or manually if needed.
     * <p>
     * This is the ONLY place where static resources are released —
     * individual {@link #close()} calls do NOT touch them.
     */
    private static void releaseStaticBatchResources() {
        synchronized (batchBufferLock) {
            if (sharedInt4BatchShader != null) {
                sharedInt4BatchShader.close();
                sharedInt4BatchShader = null;
            }
            if (sharedFp32BatchShader != null) {
                sharedFp32BatchShader.close();
                sharedFp32BatchShader = null;
            }

            // Unmap + release shared scratch buffers
            if (sharedInt4BatchUploadBuf != null) {
                D3D12Bindings.unmapResource(sharedInt4BatchUploadBuf);
                sharedInt4BatchUploadBuf = null;
            }
            if (sharedInt4BatchReadbackBuf != null) {
                D3D12Bindings.unmapResource(sharedInt4BatchReadbackBuf);
                sharedInt4BatchReadbackBuf = null;
            }
            if (sharedInt4BatchInputBuf != null) {
                DxgiBindings.release(sharedInt4BatchInputBuf);
                sharedInt4BatchInputBuf = null;
            }
            if (sharedInt4BatchOutputBuf != null) {
                DxgiBindings.release(sharedInt4BatchOutputBuf);
                sharedInt4BatchOutputBuf = null;
            }
            if (sharedInt4BatchUploadBuf != null) {
                DxgiBindings.release(sharedInt4BatchUploadBuf);
                sharedInt4BatchUploadBuf = null;
            }
            if (sharedInt4BatchReadbackBuf != null) {
                DxgiBindings.release(sharedInt4BatchReadbackBuf);
                sharedInt4BatchReadbackBuf = null;
            }

            sharedInt4BatchMappedUpload = null;
            sharedInt4BatchMappedReadback = null;
            sharedBatchCapacityM = 0;
            sharedBatchMaxK = 0;
            sharedBatchMaxN = 0;
            sharedBatchDeviceAddr = 0L;

            // Close static arena (releases all GPU resources allocated in it)
            if (sharedBatchArena != null) {
                sharedBatchArena.close();
                sharedBatchArena = null;
            }

            log.info("Static batch resources released");
        }
    }
}

