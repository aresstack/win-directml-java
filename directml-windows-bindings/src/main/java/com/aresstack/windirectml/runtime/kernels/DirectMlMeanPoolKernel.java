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
 * DirectML-Mean-Pooling über die Sequenz-Achse, formuliert als
 * Reihen-Vektor·Matrix-Produkt über {@code DML_OPERATOR_GEMM} (Op 54, FL 1.0):
 * <pre>
 *   y[1, H] = w[1, S] · x[S, H]
 * </pre>
 * <p>
 * Der {@code attentionMask}-Eingang dieses Kernels enthält bereits die
 * <b>normalisierten Gewichte</b> {@code w[t] = m[t] / Σ_t m[t]} (Float32).
 * Damit kollabiert das Pooling auf einen einzelnen FL-1.0-Dispatch und
 * benötigt weder Reduce- noch Divide-Operatoren, die auf älteren
 * In-Box-{@code DirectML.dll}-Builds (FL 5.0) nicht garantiert vorhanden
 * sind. Die Normalisierung auf der CPU kostet {@code S} Floats pro Inferenz
 * und ersetzt damit den vorher nötigen {@code [B,H]}-Read-Back nach jedem
 * Encoder-Lauf durch einen reinen {@code [H]}-Download.
 * <p>
 * Lebenszyklus: Form-gebunden auf {@code (S, H)}. Compile/Initialize einmal
 * im Konstruktor; jede {@link #dispatch} baut eine frische Binding-Table
 * und feuert eine eigene Command-List. {@link #close()} gibt alle
 * persistenten DirectML-Ressourcen frei.
 */
public final class DirectMlMeanPoolKernel implements MeanPoolingKernel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlMeanPoolKernel.class);

    private final WindowsBindings wb;
    private final int S;
    private final int H;
    private final Arena arena;

    private final MemorySegment compiled;
    private final MemorySegment cmdRecorder;
    private final MemorySegment descriptorHeap;
    private final int descriptorCount;
    private final long tempSize;
    private final long persistSize;
    private final MemorySegment tempBuffer;     // may be NULL
    private final MemorySegment persistBuffer;  // may be NULL

    private boolean closed = false;

    public DirectMlMeanPoolKernel(DirectMlContextImpl ctx, int seq, int hidden)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (seq <= 0 || hidden <= 0) {
            throw new IllegalArgumentException("seq, hidden must be > 0");
        }
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.S = seq;
        this.H = hidden;
        this.arena = Arena.ofShared();

        try {
            // GEMM: A[1,1,1,S] · B[1,1,S,H] = Y[1,1,1,H], no transpose, no bias.
            MemorySegment aDesc = bufferTensorDesc(new int[]{1, 1, 1, S},
                    (long) S * Float.BYTES);
            MemorySegment bDesc = bufferTensorDesc(new int[]{1, 1, S, H},
                    (long) S * H * Float.BYTES);
            MemorySegment yDesc = bufferTensorDesc(new int[]{1, 1, 1, H},
                    (long) H * Float.BYTES);

            // DML_GEMM_OPERATOR_DESC (56 bytes, see DirectMlLinearKernel).
            MemorySegment gemm = arena.allocate(56, 8);
            gemm.set(ValueLayout.ADDRESS, 0, aDesc);
            gemm.set(ValueLayout.ADDRESS, 8, bDesc);
            gemm.set(ValueLayout.ADDRESS, 16, MemorySegment.NULL);      // CTensor
            gemm.set(ValueLayout.ADDRESS, 24, yDesc);
            gemm.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
            gemm.set(ValueLayout.JAVA_INT, 36, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
            gemm.set(ValueLayout.JAVA_FLOAT, 40, 1.0f);                 // Alpha
            gemm.set(ValueLayout.JAVA_FLOAT, 44, 0.0f);                 // Beta
            gemm.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL);      // FusedActivation

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

            log.info("DirectMlMeanPoolKernel ready: S={}, H={}, desc={}, temp={}B, persist={}B",
                    S, H, descriptorCount, tempSize, persistSize);
        } catch (WindowsNativeException e) {
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlMeanPoolKernel", e);
        } catch (RuntimeException e) {
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlMeanPoolKernel", e);
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
     * Dispatch the pooling. {@code attentionMask} must contain the
     * pre-normalised per-token weights {@code m[t]/Σm} (Float32, length {@code S}).
     */
    @Override
    public void dispatch(DirectMlTensor tokenEmbeddings,
                         DirectMlTensor attentionMask,
                         DirectMlTensor y) throws DirectMlRuntimeException {
        ensureOpen();
        validate2D(tokenEmbeddings, S, H, "tokenEmbeddings");
        validate1D(attentionMask, S, "attentionMask");
        validate1D(y, H, "y");

        DefaultGpuBuffer aBuf = unwrap(attentionMask.buffer(), "attentionMask");
        DefaultGpuBuffer bBuf = unwrap(tokenEmbeddings.buffer(), "tokenEmbeddings");
        DefaultGpuBuffer yBuf = unwrap(y.buffer(), "y");

        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q   = wb.getCommandQueue();

        try (Arena scratch = Arena.ofConfined()) {
            long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, scratch);
            long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, scratch);
            MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, compiled,
                    cpuStart, gpuStart, descriptorCount);
            MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);
            try {
                // GEMM has 3 input slots (A, B, C). C is bound as NONE.
                MemorySegment inputs = scratch.allocate(16L * 3, 8);
                setBufferBinding(scratch, inputs, 0, aBuf.resource(), (long) S * Float.BYTES);
                setBufferBinding(scratch, inputs, 1, bBuf.resource(), (long) S * H * Float.BYTES);
                inputs.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_BINDING_TYPE_NONE);
                inputs.set(ValueLayout.ADDRESS, 40, MemorySegment.NULL);
                DirectMlBindings.bindInputs(bt, 3, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, yBuf.resource(), (long) H * Float.BYTES);
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
                    D3D12Bindings.executeOrDefer(dev, q, cl, alloc, scratch);
                } finally {
                    if (cl != null) DxgiBindings.release(cl);
                    DxgiBindings.release(alloc);
                }
            } finally {
                DxgiBindings.release(bt);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException("DirectMlMeanPoolKernel.dispatch failed", e);
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try { DxgiBindings.release(cmdRecorder); }    catch (Exception ignored) {}
        try { DxgiBindings.release(descriptorHeap); } catch (Exception ignored) {}
        try { DxgiBindings.release(compiled); }       catch (Exception ignored) {}
        if (!persistBuffer.equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(persistBuffer); } catch (Exception ignored) {}
        }
        if (!tempBuffer.equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(tempBuffer); }    catch (Exception ignored) {}
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
        if (closed) throw new DirectMlRuntimeException("DirectMlMeanPoolKernel already closed");
    }

    private static void validate2D(DirectMlTensor t, int d0, int d1, String name)
            throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32, was " + t.dataType());
        }
        int rank = t.shape().rank();
        int dim0 = (rank >= 2) ? t.shape().dim(rank - 2) : 1;
        int dim1 = t.shape().dim(rank - 1);
        if (dim0 != d0 || dim1 != d1) {
            throw new DirectMlRuntimeException(name + " expected shape ["
                    + d0 + "," + d1 + "], got " + t.shape());
        }
    }

    private static void validate1D(DirectMlTensor t, int n, String name)
            throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32, was " + t.dataType());
        }
        if (t.shape().elementCount() != n) {
            throw new DirectMlRuntimeException(name + " must hold " + n + " elements, got "
                    + t.shape().elementCount());
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

    public int seq()    { return S; }
    public int hidden() { return H; }
}

