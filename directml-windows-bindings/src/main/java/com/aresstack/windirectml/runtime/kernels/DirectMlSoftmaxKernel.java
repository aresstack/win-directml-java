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
 * Row-wise Softmax über {@code DML_OPERATOR_ACTIVATION_SOFTMAX} (Enum-ID 48,
 * FL 2.0). Normalisiert jede Zeile separat: {@code y_ij = exp(x_ij) /
 * sum_k exp(x_ik)}; die DirectML-Implementierung ist numerisch stabil
 * (subtrahiert intern das Zeilenmaximum).
 * <p>
 * Tensor-Shape: {@code [1, 1, numRows, rowLength]} – Softmax wird immer
 * über die innerste Dimension (die "Spalten") berechnet. Für Attention
 * sind die Score-Tensoren {@code [batch·heads, seq, seq]}: die werden
 * vor dem Dispatch zu {@code [1, 1, batch·heads·seq, seq]} flach gemacht.
 * <p>
 * Operator-Desc-Layout ({@code DML_ACTIVATION_SOFTMAX_OPERATOR_DESC},
 * 16 Bytes):
 * <pre>
 * struct {
 *     const DML_TENSOR_DESC* InputTensor;   // off 0
 *     const DML_TENSOR_DESC* OutputTensor;  // off 8
 * };
 * </pre>
 * Lebenszyklus identisch zu {@link DirectMlGeluKernel}: Compile + Init
 * einmal im Konstruktor, jeder {@code dispatch} baut eine frische
 * Binding-Table und feuert eine eigene Command-List.
 */
public final class DirectMlSoftmaxKernel implements SoftmaxKernel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlSoftmaxKernel.class);

    private final WindowsBindings wb;
    private final int numRows;
    private final int rowLength;
    private final int elementCount;
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

    /**
     * @param ctx       initialisierter Context (muss DirectML haben)
     * @param numRows   Anzahl der unabhängigen Softmax-Vektoren
     * @param rowLength Länge jeder Softmax-Reihe (Achse, über die normalisiert wird)
     */
    public DirectMlSoftmaxKernel(DirectMlContextImpl ctx, int numRows, int rowLength)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (numRows <= 0 || rowLength <= 0) {
            throw new IllegalArgumentException("numRows and rowLength must be > 0");
        }
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.numRows = numRows;
        this.rowLength = rowLength;
        this.elementCount = numRows * rowLength;
        this.arena = Arena.ofShared();

        try {
            int[] shape = { 1, 1, numRows, rowLength };
            long totalBytes = (long) elementCount * Float.BYTES;
            MemorySegment xDesc = bufferTensorDesc(shape, null, totalBytes);
            MemorySegment yDesc = bufferTensorDesc(shape, null, totalBytes);

            // DML_ACTIVATION_SOFTMAX_OPERATOR_DESC: InputTensor + OutputTensor
            MemorySegment desc = arena.allocate(16, 8);
            desc.set(ValueLayout.ADDRESS, 0, xDesc);
            desc.set(ValueLayout.ADDRESS, 8, yDesc);

            MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                    DirectMlBindings.DML_OPERATOR_ACTIVATION_SOFTMAX, desc);

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
                    ? D3D12Bindings.createDefaultBuffer(dev, tempSize, arena) : MemorySegment.NULL;
            this.persistBuffer = (persistSize > 0)
                    ? D3D12Bindings.createDefaultBuffer(dev, persistSize, arena) : MemorySegment.NULL;

            if (persistSize > 0) {
                initializeOperator();
            }

            log.info("DirectMlSoftmaxKernel ready: rows={}, rowLength={}, desc={}, temp={}B, persist={}B",
                    numRows, rowLength, descriptorCount, tempSize, persistSize);
        } catch (WindowsNativeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException(
                    "Failed to build DirectMlSoftmaxKernel" + dbg, e);
        } catch (RuntimeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException(
                    "Failed to build DirectMlSoftmaxKernel" + dbg, e);
        }
    }

    private static String formatDebugMessages(WindowsBindings wb) {
        java.util.List<String> msgs = wb.drainDebugMessages();
        if (msgs.isEmpty()) return "";
        StringBuilder sb = new StringBuilder("\n--- DirectML/D3D12 debug messages ---");
        for (String m : msgs) sb.append("\n  ").append(m);
        return sb.toString();
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
                MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, persistBuffer, 0, persistSize);
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

    @Override
    public void dispatch(DirectMlTensor x, DirectMlTensor y) throws DirectMlRuntimeException {
        ensureOpen();
        validate(x, "x");
        validate(y, "y");

        DefaultGpuBuffer xb = unwrap(x.buffer(), "x");
        DefaultGpuBuffer yb = unwrap(y.buffer(), "y");

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
                MemorySegment inputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, inputs, 0, xb.resource(), (long) elementCount * Float.BYTES);
                DirectMlBindings.bindInputs(bt, 1, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, yb.resource(), (long) elementCount * Float.BYTES);
                DirectMlBindings.bindOutputs(bt, 1, outputs);

                if (tempSize > 0) {
                    MemorySegment binding = DirectMlBindings.allocBufferBinding(scratch,
                            tempBuffer, 0, tempSize);
                    DirectMlBindings.bindTemporaryResource(bt,
                            DirectMlBindings.allocBindingDesc(scratch,
                                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, binding));
                }
                if (persistSize > 0) {
                    MemorySegment binding = DirectMlBindings.allocBufferBinding(scratch,
                            persistBuffer, 0, persistSize);
                    DirectMlBindings.bindPersistentResource(bt,
                            DirectMlBindings.allocBindingDesc(scratch,
                                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, binding));
                }

                MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
                MemorySegment cl = null;
                try {
                    cl = D3D12Bindings.createCommandList(dev,
                            D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);
                    transitionToUav(cl, xb, scratch);
                    transitionToUav(cl, yb, scratch);
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
            throw new DirectMlRuntimeException(
                    "DirectMlSoftmaxKernel.dispatch failed" + formatDebugMessages(wb), e);
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try { DxgiBindings.release(cmdRecorder); } catch (Exception ignored) {}
        try { DxgiBindings.release(descriptorHeap); } catch (Exception ignored) {}
        try { DxgiBindings.release(compiled); } catch (Exception ignored) {}
        if (!persistBuffer.equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(persistBuffer); } catch (Exception ignored) {}
        }
        if (!tempBuffer.equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(tempBuffer); } catch (Exception ignored) {}
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
        if (closed) throw new DirectMlRuntimeException("DirectMlSoftmaxKernel already closed");
    }

    private void validate(DirectMlTensor t, String name) throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32, was " + t.dataType());
        }
        if (t.shape().elementCount() != elementCount) {
            throw new DirectMlRuntimeException(name + " must hold " + elementCount
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

    private MemorySegment bufferTensorDesc(int[] sizes, int[] strides, long totalSizeBytes) {
        MemorySegment buf = DirectMlBindings.allocBufferTensorDesc(arena,
                DirectMlBindings.DML_TENSOR_DATA_TYPE_FLOAT32, sizes, strides, totalSizeBytes);
        return DirectMlBindings.allocTensorDesc(arena, buf);
    }

    private static void setBufferBinding(Arena scratch, MemorySegment array, int index,
                                         MemorySegment resource, long size) {
        MemorySegment bb = DirectMlBindings.allocBufferBinding(scratch, resource, 0, size);
        long off = (long) index * 16;
        array.set(ValueLayout.JAVA_INT, off, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
        array.set(ValueLayout.ADDRESS, off + 8, bb);
    }

    public int numRows() { return numRows; }
    public int rowLength() { return rowLength; }
}

