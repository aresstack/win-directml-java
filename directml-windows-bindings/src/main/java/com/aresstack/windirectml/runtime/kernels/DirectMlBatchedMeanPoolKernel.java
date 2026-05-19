package com.aresstack.windirectml.runtime.kernels;

import com.aresstack.windirectml.runtime.DefaultGpuBuffer;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.runtime.DirectMlRuntimeException;
import com.aresstack.windirectml.runtime.DirectMlTensor;
import com.aresstack.windirectml.runtime.GpuBuffer;
import com.aresstack.windirectml.runtime.TensorDataType;
import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.DirectMlBindings;
import com.aresstack.windirectml.windows.DxgiBindings;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Batched DirectML mean-pooling: {@code N} independent
 * {@code [1,S]·[S,H] = [1,H]} GEMMs in a single dispatch.
 * <p>
 * Tensor shapes:
 * <pre>
 *   weights w : [N, 1, 1, S]   (pre-normalised m[t]/Σm per row)
 *   hidden  x : [N, 1, S, H]
 *   pooled  y : [N, 1, 1, H]
 * </pre>
 * DirectML's {@code DML_OPERATOR_GEMM} treats the leading two dimensions
 * as batch axes, so a single {@code DML_GEMM_OPERATOR_DESC} computes
 * <pre>
 *   y[n, h] = Σ_t w[n, t] · x[n, t, h]
 * </pre>
 * for all {@code n ∈ [0, N)} in one shot. FL-1.0 compatible.
 * <p>
 * The constructor compiles + initialises the form-bound graph for
 * exactly one {@code (N, S, H)}. {@link #close()} releases all
 * persistent DirectML resources. Each {@link #dispatch} builds a fresh
 * binding table and submits a self-contained command list.
 */
public final class DirectMlBatchedMeanPoolKernel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlBatchedMeanPoolKernel.class);

    private final WindowsBindings wb;
    private final int N;
    private final int S;
    private final int H;
    private final Arena arena;

    private final MemorySegment compiled;
    private final MemorySegment cmdRecorder;
    private final MemorySegment descriptorHeap;
    private final int descriptorCount;
    private final long tempSize;
    private final long persistSize;
    private final MemorySegment tempBuffer;
    private final MemorySegment persistBuffer;

    private boolean closed = false;

    public DirectMlBatchedMeanPoolKernel(DirectMlContextImpl ctx, int batch, int seq, int hidden)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (batch <= 0 || seq <= 0 || hidden <= 0) {
            throw new IllegalArgumentException("batch, seq, hidden must be > 0");
        }
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.N = batch;
        this.S = seq;
        this.H = hidden;
        this.arena = Arena.ofShared();

        try {
            long aBytes = (long) N * S * Float.BYTES;
            long bBytes = (long) N * S * H * Float.BYTES;
            long yBytes = (long) N * H * Float.BYTES;

            MemorySegment aDesc = bufferTensorDesc(new int[]{N, 1, 1, S}, aBytes);
            MemorySegment bDesc = bufferTensorDesc(new int[]{N, 1, S, H}, bBytes);
            MemorySegment yDesc = bufferTensorDesc(new int[]{N, 1, 1, H}, yBytes);

            // DML_GEMM_OPERATOR_DESC (56 bytes – same layout as DirectMlMeanPoolKernel).
            MemorySegment gemm = arena.allocate(56, 8);
            gemm.set(ValueLayout.ADDRESS, 0, aDesc);
            gemm.set(ValueLayout.ADDRESS, 8, bDesc);
            gemm.set(ValueLayout.ADDRESS, 16, MemorySegment.NULL);                   // CTensor
            gemm.set(ValueLayout.ADDRESS, 24, yDesc);
            gemm.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
            gemm.set(ValueLayout.JAVA_INT, 36, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
            gemm.set(ValueLayout.JAVA_FLOAT, 40, 1.0f);                              // Alpha
            gemm.set(ValueLayout.JAVA_FLOAT, 44, 0.0f);                              // Beta
            gemm.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL);                   // FusedActivation

            MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                    DirectMlBindings.DML_OPERATOR_GEMM, gemm);
            MemorySegment dml = wb.getDmlDevice();
            MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
            this.compiled = DirectMlBindings.compileOperator(dml, op,
                    DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
            DxgiBindings.release(op);

            long[] bp = DirectMlBindings.getBindingProperties(compiled, arena);
            this.descriptorCount = Math.max((int) bp[0], 1);
            this.tempSize = bp[1];
            this.persistSize = bp[2];

            MemorySegment dev = wb.getD3d12Device();
            this.descriptorHeap = D3D12Bindings.createDescriptorHeap(dev, descriptorCount, arena);
            this.cmdRecorder = DirectMlBindings.createCommandRecorder(dml, arena);

            this.tempBuffer = (tempSize > 0)
                    ? D3D12Bindings.createDefaultBuffer(dev, tempSize, arena)
                    : MemorySegment.NULL;
            this.persistBuffer = (persistSize > 0)
                    ? D3D12Bindings.createDefaultBuffer(dev, persistSize, arena)
                    : MemorySegment.NULL;

            if (persistSize > 0) {
                initializeOperator();
            }

            log.info("DirectMlBatchedMeanPoolKernel ready: N={}, S={}, H={}, desc={}, temp={}B, persist={}B",
                    N, S, H, descriptorCount, tempSize, persistSize);
        } catch (WindowsNativeException e) {
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlBatchedMeanPoolKernel", e);
        } catch (RuntimeException e) {
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlBatchedMeanPoolKernel", e);
        }
    }

    private void initializeOperator() throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();

        MemorySegment initializer = DirectMlBindings.createOperatorInitializer(dml,
                new MemorySegment[]{compiled}, arena);
        long[] ibp = DirectMlBindings.getBindingProperties(initializer, arena);
        int initDescCount = Math.max((int) ibp[0], 1);
        long initTempSize = ibp[1];

        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, initializer,
                cpuStart, gpuStart, initDescCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);
        try {
            MemorySegment inputBindings = DirectMlBindings.allocNoneBindingDesc(arena);
            DirectMlBindings.bindInputs(bt, 1, inputBindings);

            if (persistSize > 0) {
                MemorySegment outArr = arena.allocate(16, 8);
                MemorySegment bb = DirectMlBindings.allocBufferBinding(arena,
                        persistBuffer, 0, persistSize);
                outArr.set(ValueLayout.JAVA_INT, 0, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
                outArr.set(ValueLayout.ADDRESS, 8, bb);
                DirectMlBindings.bindOutputs(bt, 1, outArr);
            }

            MemorySegment initTmp = MemorySegment.NULL;
            if (initTempSize > 0) {
                initTmp = D3D12Bindings.createDefaultBuffer(dev, initTempSize, arena);
                MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, initTmp, 0, initTempSize);
                MemorySegment bd = DirectMlBindings.allocBindingDesc(arena,
                        DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
                DirectMlBindings.bindTemporaryResource(bt, bd);
            }

            MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
            MemorySegment cl = null;
            try {
                cl = D3D12Bindings.createCommandList(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, arena);
                D3D12Bindings.setDescriptorHeaps(cl, descriptorHeap, arena);
                DirectMlBindings.recordDispatch(cmdRecorder, cl, initializer, bt);
                D3D12Bindings.executeAndWait(dev, q, cl, arena);
            } finally {
                if (cl != null) DxgiBindings.release(cl);
                DxgiBindings.release(alloc);
                if (!initTmp.equals(MemorySegment.NULL)) DxgiBindings.release(initTmp);
            }
        } finally {
            DxgiBindings.release(bt);
            DxgiBindings.release(initializer);
        }
    }

    /**
     * Dispatch one batched mean-pool. {@code weights} must contain the
     * pre-normalised per-token weights {@code w[n,t] = m[n,t]/Σ_t m[n,t]}
     * (Float32, total {@code N·S} elements).
     */
    public void dispatch(DirectMlTensor hidden,
                         DirectMlTensor weights,
                         DirectMlTensor pooled) throws DirectMlRuntimeException {
        ensureOpen();
        validateElements(hidden, (long) N * S * H, "hidden");
        validateElements(weights, (long) N * S, "weights");
        validateElements(pooled, (long) N * H, "pooled");

        DefaultGpuBuffer aBuf = unwrap(weights.buffer(), "weights");
        DefaultGpuBuffer bBuf = unwrap(hidden.buffer(), "hidden");
        DefaultGpuBuffer yBuf = unwrap(pooled.buffer(), "pooled");

        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();

        try (Arena scratch = Arena.ofConfined()) {
            long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, scratch);
            long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, scratch);
            MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, compiled,
                    cpuStart, gpuStart, descriptorCount);
            MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);
            try {
                MemorySegment inputs = scratch.allocate(16L * 3, 8);
                setBufferBinding(scratch, inputs, 0, aBuf.resource(), (long) N * S * Float.BYTES);
                setBufferBinding(scratch, inputs, 1, bBuf.resource(), (long) N * S * H * Float.BYTES);
                inputs.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_BINDING_TYPE_NONE);
                inputs.set(ValueLayout.ADDRESS, 40, MemorySegment.NULL);
                DirectMlBindings.bindInputs(bt, 3, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, yBuf.resource(), (long) N * H * Float.BYTES);
                DirectMlBindings.bindOutputs(bt, 1, outputs);

                if (tempSize > 0) {
                    MemorySegment bbTmp = DirectMlBindings.allocBufferBinding(scratch,
                            tempBuffer, 0, tempSize);
                    DirectMlBindings.bindTemporaryResource(bt,
                            DirectMlBindings.allocBindingDesc(scratch,
                                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bbTmp));
                }
                if (persistSize > 0) {
                    MemorySegment bbPer = DirectMlBindings.allocBufferBinding(scratch,
                            persistBuffer, 0, persistSize);
                    DirectMlBindings.bindPersistentResource(bt,
                            DirectMlBindings.allocBindingDesc(scratch,
                                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bbPer));
                }

                MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
                MemorySegment cl = null;
                try {
                    cl = D3D12Bindings.createCommandList(dev,
                            D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);
                    transitionToUav(cl, aBuf, scratch);
                    transitionToUav(cl, bBuf, scratch);
                    transitionToUav(cl, yBuf, scratch);
                    D3D12Bindings.setDescriptorHeaps(cl, descriptorHeap, scratch);
                    DirectMlBindings.recordDispatch(cmdRecorder, cl, compiled, bt);
                    D3D12Bindings.executeAndWait(dev, q, cl, scratch);
                } finally {
                    if (cl != null) DxgiBindings.release(cl);
                    DxgiBindings.release(alloc);
                }
            } finally {
                DxgiBindings.release(bt);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException("DirectMlBatchedMeanPoolKernel.dispatch failed", e);
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            DxgiBindings.release(cmdRecorder);
        } catch (Exception ignored) {
        }
        try {
            DxgiBindings.release(descriptorHeap);
        } catch (Exception ignored) {
        }
        try {
            DxgiBindings.release(compiled);
        } catch (Exception ignored) {
        }
        if (!persistBuffer.equals(MemorySegment.NULL)) {
            try {
                DxgiBindings.release(persistBuffer);
            } catch (Exception ignored) {
            }
        }
        if (!tempBuffer.equals(MemorySegment.NULL)) {
            try {
                DxgiBindings.release(tempBuffer);
            } catch (Exception ignored) {
            }
        }
        arena.close();
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private static void transitionToUav(MemorySegment cl, DefaultGpuBuffer buf, Arena scratch) {
        int state = buf.currentResourceState();
        if (state != D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12Bindings.transitionBarrier(cl, buf.resource(), state,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS, scratch);
            buf.setCurrentResourceState(D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }

    private void ensureOpen() throws DirectMlRuntimeException {
        if (closed) throw new DirectMlRuntimeException("DirectMlBatchedMeanPoolKernel already closed");
    }

    private static void validateElements(DirectMlTensor t, long expected, String name)
            throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32, was " + t.dataType());
        }
        if (t.shape().elementCount() != expected) {
            throw new DirectMlRuntimeException(name + " must hold " + expected
                    + " elements, got " + t.shape().elementCount());
        }
    }

    private static DefaultGpuBuffer unwrap(GpuBuffer buf, String name) throws DirectMlRuntimeException {
        if (!(buf instanceof DefaultGpuBuffer d)) {
            throw new DirectMlRuntimeException(name + " buffer must be DefaultGpuBuffer (got "
                    + (buf == null ? "null" : buf.getClass().getName()) + ")");
        }
        return d;
    }

    private MemorySegment bufferTensorDesc(int[] sizes, long totalSizeBytes) {
        MemorySegment buf = DirectMlBindings.allocBufferTensorDesc(arena,
                DirectMlBindings.DML_TENSOR_DATA_TYPE_FLOAT32, sizes, null, totalSizeBytes);
        return DirectMlBindings.allocTensorDesc(arena, buf);
    }

    private static void setBufferBinding(Arena scratch, MemorySegment array, int index,
                                         MemorySegment resource, long size) {
        MemorySegment bb = DirectMlBindings.allocBufferBinding(scratch, resource, 0, size);
        long off = (long) index * 16;
        array.set(ValueLayout.JAVA_INT, off, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
        array.set(ValueLayout.ADDRESS, off + 8, bb);
    }

    public int batch() {
        return N;
    }

    public int seq() {
        return S;
    }

    public int hidden() {
        return H;
    }
}

