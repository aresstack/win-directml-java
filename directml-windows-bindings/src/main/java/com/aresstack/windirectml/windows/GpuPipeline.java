package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Shared GPU command pipeline for batching multiple DML dispatches into a single
 * D3D12 command list submission.
 * <p>
 * <b>V2.0 submission collapse</b>: Instead of each {@link MatMulNBitsKernel} owning its own
 * command allocator, command list, and fence (129 separate submissions per token), this
 * pipeline provides shared infrastructure. Multiple DML dispatches, barrier transitions,
 * and buffer copies can be recorded into ONE command list and executed with ONE fence wait.
 * <p>
 * The pipeline also manages GPU-resident activation buffers that stay on the GPU between
 * operations, eliminating the CPU roundtrip for intermediate results.
 * <p>
 * <b>Lifecycle</b>:
 * <ol>
 *   <li>{@link #begin()} — reset allocator + command list, start recording</li>
 *   <li>Record operations: {@link #recordUpload}, {@link #recordDispatch},
 *       {@link #recordBarrier}, {@link #recordCopy}, {@link #recordReadback}</li>
 *   <li>{@link #submitAndWait()} — close + execute + fence wait</li>
 * </ol>
 *
 * @see MatMulNBitsKernel#recordInto(GpuPipeline)
 */
public final class GpuPipeline implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(GpuPipeline.class);

    private final WindowsBindings wb;
    private final Arena arena;

    // ── Shared command infrastructure ──────────────────────────────────
    private MemorySegment cmdAllocator;
    private MemorySegment cmdList;
    private MemorySegment fence;
    private long fenceValue;

    // ── Shared staging buffers (sized for max activation) ─────────────
    // Upload buffer: CPU → GPU (persistently mapped)
    private MemorySegment uploadBuf;
    private MemorySegment mappedUpload;
    private long uploadBufSize;

    // Readback buffer: GPU → CPU (persistently mapped)
    private MemorySegment readbackBuf;
    private MemorySegment mappedReadback;
    private long readbackBufSize;

    // ── Pre-cached MethodHandles (zero-alloc hot path) ────────────────
    private MethodHandle mhResetAllocator;
    private MethodHandle mhResetCmdList;
    private MethodHandle mhCopyBufferRegion;
    private MethodHandle mhResourceBarrier;
    private MethodHandle mhSetDescriptorHeaps;
    private MethodHandle mhCloseCmdList;
    private MethodHandle mhExecuteCmdLists;
    private MethodHandle mhQueueSignal;
    private MethodHandle mhFenceGetCompleted;

    // ── Command list array for ExecuteCommandLists ─────────────────────
    private MemorySegment cmdListArrayPtr;

    private boolean recording = false;
    private boolean closed = false;

    /**
     * Create a GPU pipeline with shared command infrastructure.
     *
     * @param wb               initialized WindowsBindings
     * @param maxUploadBytes   size of the shared upload buffer (largest input activation)
     * @param maxReadbackBytes size of the shared readback buffer (largest output activation)
     */
    public GpuPipeline(WindowsBindings wb, long maxUploadBytes, long maxReadbackBytes)
            throws WindowsNativeException {
        this.wb = wb;
        this.arena = Arena.ofShared();
        this.uploadBufSize = maxUploadBytes;
        this.readbackBufSize = maxReadbackBytes;

        var dev = wb.getD3d12Device();
        var queue = wb.getCommandQueue();

        // ── Shared command allocator + list ────────────────────────────
        cmdAllocator = D3D12Bindings.createCommandAllocator(dev,
                D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        cmdList = D3D12Bindings.createCommandList(dev,
                D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAllocator, arena);
        D3D12Bindings.closeCommandList(cmdList); // close so we can Reset it later

        // ── Shared fence ──────────────────────────────────────────────
        fence = D3D12Bindings.createFence(dev, 0, arena);
        fenceValue = 0;

        // ── Shared staging buffers ────────────────────────────────────
        uploadBuf = D3D12Bindings.createUploadBuffer(dev, maxUploadBytes, arena);
        readbackBuf = D3D12Bindings.createReadbackBuffer(dev, maxReadbackBytes, arena);
        mappedUpload = D3D12Bindings.mapResource(uploadBuf, arena);
        mappedReadback = D3D12Bindings.mapResource(readbackBuf, arena);

        // ── Command list array ────────────────────────────────────────
        cmdListArrayPtr = arena.allocate(ValueLayout.ADDRESS);
        cmdListArrayPtr.set(ValueLayout.ADDRESS, 0, cmdList);

        // ── Pre-cache MethodHandles ───────────────────────────────────
        mhResetAllocator = DxgiBindings.vtableMethod(cmdAllocator, 8,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhResetCmdList = DxgiBindings.vtableMethod(cmdList, 10,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhCopyBufferRegion = DxgiBindings.vtableMethod(cmdList, 15,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_LONG));
        mhResourceBarrier = DxgiBindings.vtableMethod(cmdList, 26,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhSetDescriptorHeaps = DxgiBindings.vtableMethod(cmdList, 28,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhCloseCmdList = DxgiBindings.vtableMethod(cmdList, 9,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhExecuteCmdLists = DxgiBindings.vtableMethod(queue, 10,
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhQueueSignal = DxgiBindings.vtableMethod(queue, 14,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        mhFenceGetCompleted = DxgiBindings.vtableMethod(fence, 8,
                FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

        log.info("GpuPipeline created: upload={}KB, readback={}KB",
                maxUploadBytes / 1024, maxReadbackBytes / 1024);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Recording API
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Begin recording a new command batch. Resets the shared allocator and
     * command list. Must be called before any record* methods.
     */
    public void begin() {
        if (recording) throw new IllegalStateException("Already recording — call submitAndWait first");
        try {
            int hr = (int) mhResetAllocator.invokeExact(cmdAllocator);
            HResult.check(hr, "Pipeline::ResetAllocator");
            hr = (int) mhResetCmdList.invokeExact(cmdList, cmdAllocator, MemorySegment.NULL);
            HResult.check(hr, "Pipeline::ResetCmdList");
            recording = true;
        } catch (Throwable t) {
            throw new RuntimeException("GpuPipeline.begin() failed", t);
        }
    }

    /**
     * Upload float data from a Java array into the shared upload buffer,
     * then record a copy from upload buffer → target GPU default buffer.
     *
     * @param data         source CPU array
     * @param offset       offset in source array (elements)
     * @param length       number of floats to upload
     * @param targetBuf    GPU default buffer (destination)
     * @param targetOffset byte offset in target buffer
     */
    public void recordUpload(float[] data, int offset, int length,
                             MemorySegment targetBuf, long targetOffset) {
        checkRecording();
        long bytes = (long) length * Float.BYTES;
        if (bytes > uploadBufSize)
            throw new IllegalArgumentException("Upload size " + bytes + " > buffer " + uploadBufSize);
        try {
            MemorySegment.copy(data, offset, mappedUpload, ValueLayout.JAVA_FLOAT, 0, length);
            mhCopyBufferRegion.invokeExact(cmdList, targetBuf, targetOffset, uploadBuf, 0L, bytes);
        } catch (Throwable t) {
            throw new RuntimeException("recordUpload failed", t);
        }
    }

    /**
     * Record a transition barrier on a resource.
     */
    public void recordBarrier(MemorySegment barrierStruct) {
        checkRecording();
        try {
            mhResourceBarrier.invokeExact(cmdList, 1, barrierStruct);
        } catch (Throwable t) {
            throw new RuntimeException("recordBarrier failed", t);
        }
    }

    /**
     * Record a UAV barrier (no specific resource — global sync between dispatches).
     */
    public void recordUavBarrier(MemorySegment uavBarrierStruct) {
        recordBarrier(uavBarrierStruct);
    }

    /**
     * Record setting descriptor heaps on the command list.
     */
    public void recordSetDescriptorHeaps(MemorySegment heapArrayPtr) {
        checkRecording();
        try {
            mhSetDescriptorHeaps.invokeExact(cmdList, 1, heapArrayPtr);
        } catch (Throwable t) {
            throw new RuntimeException("recordSetDescriptorHeaps failed", t);
        }
    }

    /**
     * Record a DML dispatch (GEMM or other operator).
     *
     * @param cmdRecorder      IDMLCommandRecorder
     * @param compiledOp       IDMLCompiledOperator
     * @param bindingTable     IDMLBindingTable
     * @param mhRecordDispatch pre-cached MethodHandle for RecordDispatch
     */
    public void recordDmlDispatch(MemorySegment cmdRecorder, MemorySegment compiledOp,
                                  MemorySegment bindingTable, MethodHandle mhRecordDispatch) {
        checkRecording();
        try {
            mhRecordDispatch.invokeExact(cmdRecorder, cmdList, compiledOp, bindingTable);
        } catch (Throwable t) {
            throw new RuntimeException("recordDmlDispatch failed", t);
        }
    }

    /**
     * Record a GPU-to-GPU buffer copy.
     */
    public void recordCopy(MemorySegment dst, long dstOffset,
                           MemorySegment src, long srcOffset, long numBytes) {
        checkRecording();
        try {
            mhCopyBufferRegion.invokeExact(cmdList, dst, dstOffset, src, srcOffset, numBytes);
        } catch (Throwable t) {
            throw new RuntimeException("recordCopy failed", t);
        }
    }

    /**
     * Record a copy from GPU default buffer → shared readback buffer.
     *
     * @param srcBuf    GPU source buffer
     * @param srcOffset byte offset in source
     * @param numBytes  bytes to read back
     */
    public void recordReadback(MemorySegment srcBuf, long srcOffset, long numBytes) {
        checkRecording();
        if (numBytes > readbackBufSize)
            throw new IllegalArgumentException("Readback size " + numBytes + " > buffer " + readbackBufSize);
        try {
            mhCopyBufferRegion.invokeExact(cmdList, readbackBuf, 0L, srcBuf, srcOffset, numBytes);
        } catch (Throwable t) {
            throw new RuntimeException("recordReadback failed", t);
        }
    }

    /**
     * Close the command list, submit to the GPU queue, and spin-wait for completion.
     * After this call, any readback data is available in the mapped readback buffer.
     */
    public void submitAndWait() {
        if (!recording) throw new IllegalStateException("Not recording — call begin() first");
        recording = false;
        try {
            // Close command list
            int hr = (int) mhCloseCmdList.invokeExact(cmdList);
            HResult.check(hr, "Pipeline::CloseCmdList");

            // Execute
            mhExecuteCmdLists.invokeExact(wb.getCommandQueue(), 1, cmdListArrayPtr);

            // Signal fence
            fenceValue++;
            hr = (int) mhQueueSignal.invokeExact(wb.getCommandQueue(), fence, fenceValue);
            HResult.check(hr, "Pipeline::QueueSignal");

            // Spin-wait
            long deadline = System.currentTimeMillis() + 120_000;
            while ((long) mhFenceGetCompleted.invokeExact(fence) < fenceValue) {
                if (System.currentTimeMillis() > deadline) {
                    throw new WindowsNativeException(
                            "GPU pipeline fence timeout after 120000 ms – the GPU may be hung");
                }
                Thread.onSpinWait();
            }
        } catch (WindowsNativeException e) {
            throw new RuntimeException("Pipeline submit failed", e);
        } catch (Throwable t) {
            throw new RuntimeException("Pipeline submit failed", t);
        }
    }

    /**
     * Copy float data from the readback buffer into a Java array.
     * Must be called after {@link #submitAndWait()}.
     *
     * @param out    destination array
     * @param offset offset in destination (elements)
     * @param length number of floats to read
     */
    public void readbackInto(float[] out, int offset, int length) {
        if (recording) throw new IllegalStateException("Still recording — call submitAndWait first");
        MemorySegment.copy(mappedReadback, ValueLayout.JAVA_FLOAT, 0, out, offset, length);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pre-allocated barrier helpers
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Allocate a reusable transition barrier struct in this pipeline's arena.
     */
    public MemorySegment allocTransitionBarrier(MemorySegment resource, int before, int after) {
        MemorySegment b = arena.allocate(32, 8);
        b.set(ValueLayout.JAVA_INT, 0, D3D12Bindings.D3D12_RESOURCE_BARRIER_TYPE_TRANSITION);
        b.set(ValueLayout.JAVA_INT, 4, D3D12Bindings.D3D12_RESOURCE_BARRIER_FLAG_NONE);
        b.set(ValueLayout.ADDRESS, 8, resource);
        b.set(ValueLayout.JAVA_INT, 16, D3D12Bindings.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
        b.set(ValueLayout.JAVA_INT, 20, before);
        b.set(ValueLayout.JAVA_INT, 24, after);
        return b;
    }

    /**
     * Allocate a reusable UAV barrier struct (global synchronization between DML dispatches).
     */
    public MemorySegment allocUavBarrier() {
        MemorySegment b = arena.allocate(32, 8);
        b.set(ValueLayout.JAVA_INT, 0, D3D12Bindings.D3D12_RESOURCE_BARRIER_TYPE_UAV);
        b.set(ValueLayout.JAVA_INT, 4, D3D12Bindings.D3D12_RESOURCE_BARRIER_FLAG_NONE);
        b.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL); // all UAV resources
        return b;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Accessors
    // ═══════════════════════════════════════════════════════════════════

    /**
     * The shared command list (for kernels that need to record into it).
     */
    public MemorySegment getCommandList() {
        return cmdList;
    }

    /**
     * The pipeline's arena (for allocating GPU resources that live as long as the pipeline).
     */
    public Arena getArena() {
        return arena;
    }

    /**
     * Whether the pipeline is currently recording.
     */
    public boolean isRecording() {
        return recording;
    }

    // ═══════════════════════════════════════════════════════════════════
    // AutoCloseable
    // ═══════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        if (uploadBuf != null) D3D12Bindings.unmapResource(uploadBuf);
        if (readbackBuf != null) D3D12Bindings.unmapResource(readbackBuf);

        if (fence != null) DxgiBindings.release(fence);
        if (cmdList != null) DxgiBindings.release(cmdList);
        if (cmdAllocator != null) DxgiBindings.release(cmdAllocator);
        if (readbackBuf != null) DxgiBindings.release(readbackBuf);
        if (uploadBuf != null) DxgiBindings.release(uploadBuf);

        arena.close();
        log.info("GpuPipeline closed");
    }

    private void checkRecording() {
        if (!recording) throw new IllegalStateException("Not recording — call begin() first");
    }
}

