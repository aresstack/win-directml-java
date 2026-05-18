package com.aresstack.windirectml.windows;

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

    /**
     * Create a MatMulNBits kernel for a specific weight matrix.
     *
     * @param wb     initialized WindowsBindings (D3D12 + DirectML devices)
     * @param N      output features (weight rows)
     * @param K      input features (weight cols)
     * @param qWeight   packed INT4 weights [N, K/blockSize, blockSize/2]
     * @param scales    per-block FP32 scales [N * K/blockSize]
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

        try {
            long t0 = System.nanoTime();
            float[] dequantized = dequantizeInt4(qWeight, scales, zeroPoints, N, K, blockSize);
            log.info("Dequantized [{}, {}] INT4→FP32 in {} ms",
                    N, K, (System.nanoTime() - t0) / 1_000_000);
            prepareGpu(dequantized);
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

    /** Package-private constructor for pre-dequantized FP32 weights. */
    private MatMulNBitsKernel(WindowsBindings wb, int N, int K, float[] fp32Weights) {
        this.wb = wb;
        this.arena = Arena.ofShared();
        this.N = N;
        this.K = K;

        try {
            prepareGpu(fp32Weights);
        } catch (WindowsNativeException e) {
            arena.close();
            throw new RuntimeException("MatMulNBitsKernel preparation failed (FP32 path)", e);
        }
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
        long inputBytes  = (long) K * Float.BYTES;        // M=1 for matvec
        long outputBytes = (long) N * Float.BYTES;
        long biasBytes   = (long) N * Float.BYTES;

        weightBuf = D3D12Bindings.createDefaultBuffer(dev, weightBytes, arena);
        biasBuf   = D3D12Bindings.createDefaultBuffer(dev, biasBytes, arena);
        inputBuf  = D3D12Bindings.createDefaultBuffer(dev, inputBytes, arena);
        outputBuf = D3D12Bindings.createDefaultBuffer(dev, outputBytes, arena);

        // ── Step 2: Upload weight data to GPU ─────────────────────────
        D3D12Bindings.uploadFloats(dev, queue, weightBuf, dequantized, arena);
        float[] zeroBias = new float[N]; // GEMM C tensor = zero bias
        D3D12Bindings.uploadFloats(dev, queue, biasBuf, zeroBias, arena);
        log.info("Uploaded weight [{}, {}] to GPU ({} MB)",
                N, K, weightBytes / (1024 * 1024));

        // ── Step 3: Create and compile DirectML GEMM operator ─────────
        MemorySegment gemm = arena.allocate(56, 8);
        gemm.set(ValueLayout.ADDRESS,  0, td(new int[]{1, 1, 1, K}));       // A: [1,1,1,K]
        gemm.set(ValueLayout.ADDRESS,  8, td(new int[]{1, 1, N, K}));       // B: [1,1,N,K]
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
        descCount   = Math.max((int) props[0], 1);
        tempSize    = props[1];
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
        uploadBuf   = D3D12Bindings.createUploadBuffer(dev, inputBytes, arena);
        readbackBuf = D3D12Bindings.createReadbackBuffer(dev, outputBytes, arena);
        mappedUpload   = D3D12Bindings.mapResource(uploadBuf, arena);
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
        long biasBytes   = (long) N * Float.BYTES;

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

    /** Allocate a reusable transition barrier struct in the kernel's arena. */
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
     * <b>V1.2 zero-alloc hot path</b>: upload, DML dispatch, and readback are
     * combined into a <b>single command list submission</b>. All resources
     * (staging buffers, command allocator, command list, fence, binding table,
     * barrier structs, and MethodHandles) are pre-allocated and reused across
     * calls. <b>Zero heap allocation, zero MethodHandle creation</b> in the
     * hot path. Only one GPU synchronization point per call.
     *
     * @param x   input vector [K]
     * @param out output vector [N] (must have length ≥ N)
     */
    public void matvec(float[] x, float[] out) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");
        if (x.length != K) throw new IllegalArgumentException(
                "Input length " + x.length + " != K=" + K);

        long inputBytes  = (long) K * Float.BYTES;
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
            //    V1.3: Removed redundant UAV barrier (the transition barrier
            //    output UAV→COPY_SOURCE already provides necessary synchronization).
            //    Cleanup barriers to COMMON remain because D3D12 only decays
            //    implicitly-promoted resources, not explicitly-transitioned ones.
            mhCopyBufferRegion.invokeExact(execCmdList, inputBuf, 0L, uploadBuf, 0L, inputBytes);
            mhResourceBarrier.invokeExact(execCmdList, 1, barrierInputToUAV);
            mhSetDescriptorHeaps.invokeExact(execCmdList, 1, heapArrayPtr);
            mhRecordDispatch.invokeExact(cmdRecorder, execCmdList, compiledGemm, execBindingTable);
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
            long deadline = System.currentTimeMillis() + 10_000;
            while ((long) mhFenceGetCompleted.invokeExact(execFence) < fenceValue) {
                if (System.currentTimeMillis() > deadline) {
                    throw new WindowsNativeException(
                            "GPU fence timeout after 10000 ms – the GPU may be hung");
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

        long inputBytes  = (long) K * Float.BYTES;
        long outputBytes = (long) N * Float.BYTES;

        try {
            // 1. Write input to persistently-mapped upload buffer (kernel-local)
            MemorySegment.copy(x, 0, mappedUpload, ValueLayout.JAVA_FLOAT, 0, K);

            // 2. Record: upload → barrier → set heaps → dispatch → barrier → readback
            var cl = pipeline.getCommandList();
            mhCopyBufferRegion.invokeExact(cl, inputBuf, 0L, uploadBuf, 0L, inputBytes);
            mhResourceBarrier.invokeExact(cl, 1, barrierInputToUAV);
            mhSetDescriptorHeaps.invokeExact(cl, 1, heapArrayPtr);
            mhRecordDispatch.invokeExact(cmdRecorder, cl, compiledGemm, execBindingTable);
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
     * Record this kernel's dispatch that reads input from a GPU-resident buffer
     * instead of uploading from CPU. Used when the activation is already on GPU
     * (e.g., output of a previous operation in the same command list).
     *
     * @param pipeline       the shared GPU pipeline (recording state)
     * @param gpuInputBuf    GPU default buffer containing the input [K] floats
     * @param gpuInputBytes  byte size of the input data
     */
    public void recordIntoGpuResident(GpuPipeline pipeline, MemorySegment gpuInputBuf,
                                       long gpuInputBytes) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");

        long outputBytes = (long) N * Float.BYTES;

        try {
            var cl = pipeline.getCommandList();
            // Copy from GPU-resident source → this kernel's input buffer
            mhCopyBufferRegion.invokeExact(cl, inputBuf, 0L, gpuInputBuf, 0L, gpuInputBytes);
            mhResourceBarrier.invokeExact(cl, 1, barrierInputToUAV);
            mhSetDescriptorHeaps.invokeExact(cl, 1, heapArrayPtr);
            mhRecordDispatch.invokeExact(cmdRecorder, cl, compiledGemm, execBindingTable);
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
            mhSetDescriptorHeaps.invokeExact(cl, 1, heapArrayPtr);
            mhRecordDispatch.invokeExact(cmdRecorder, cl, compiledGemm, execBindingTable);
            // Output stays in UAV — no readback, no cleanup
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.recordBatchFromCpu failed", t);
        }
    }

    /**
     * Record only the DML dispatch — input buffer already contains data in UAV state
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
            mhSetDescriptorHeaps.invokeExact(cl, 1, heapArrayPtr);
            mhRecordDispatch.invokeExact(cmdRecorder, cl, compiledGemm, execBindingTable);
            // Output stays in UAV
        } catch (Throwable t) {
            throw new RuntimeException("MatMulNBitsKernel.recordBatchDispatchOnly failed", t);
        }
    }

    // ── GPU buffer accessors (for pipeline-batched operations) ─────────

    /** GPU output buffer (default heap, UAV). Result is written here by DML dispatch. */
    public MemorySegment getOutputBuf() { return outputBuf; }

    /** GPU input buffer (default heap, UAV). */
    public MemorySegment getInputBuf() { return inputBuf; }

    /** Readback buffer (readback heap, mapped). */
    public MemorySegment getReadbackBuf() { return readbackBuf; }

    /** Mapped readback pointer. */
    public MemorySegment getMappedReadback() { return mappedReadback; }

    /** Mapped upload pointer. */
    public MemorySegment getMappedUpload() { return mappedUpload; }

    /** Upload buffer. */
    public MemorySegment getUploadBuf() { return uploadBuf; }

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
                    result[rowBase + kBase + 2 * j]     = w0 * scale;
                    result[rowBase + kBase + 2 * j + 1] = w1 * scale;
                }
            }
        }
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Helpers
    // ══════════════════════════════════════════════════════════════════════

    /** Build a DML_TENSOR_DESC for FP32 buffer tensor. */
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

        // Release pre-allocated execution infrastructure (reverse creation order)
        if (execBindingTable != null) DxgiBindings.release(execBindingTable);
        if (execFence != null) DxgiBindings.release(execFence);
        if (execCmdList != null) DxgiBindings.release(execCmdList);
        if (execAllocator != null) DxgiBindings.release(execAllocator);
        if (readbackBuf != null) DxgiBindings.release(readbackBuf);
        if (uploadBuf != null) DxgiBindings.release(uploadBuf);

        // Release operator resources
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
        log.debug("MatMulNBitsKernel closed [{}, {}]", N, K);
    }

    /** Output features (rows of weight matrix). */
    public int getN() { return N; }

    /** Input features (cols of weight matrix). */
    public int getK() { return K; }
}

