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
 * Element-wise sum {@code y = a + b} via {@code DML_OPERATOR_ELEMENT_WISE_ADD}
 * (op 4, FL 1.0).
 * <p>
 * Both inputs must be the same flat element count; shapes are flattened to
 * {@code [1, 1, 1, N]}. Output may alias either input (DirectML allows the
 * in-place case via the same UAV buffer for {@code a} and {@code y}).
 * <p>
 * Primary use case in this project: BERT/MiniLM residual connections
 * ({@code x = x + attnOut} and {@code x = x + mlpOut}). The
 * {@link DirectMlAttentionKernel} also embeds an ADD internally for the
 * mask, but that one uses broadcast strides and is not reusable for the
 * residual path.
 */
public final class DirectMlAddKernel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlAddKernel.class);

    private final WindowsBindings wb;
    private final int elementCount;
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

    public DirectMlAddKernel(DirectMlContextImpl ctx, int elementCount)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (elementCount <= 0) {
            throw new IllegalArgumentException("elementCount must be > 0");
        }
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.elementCount = elementCount;
        this.arena = Arena.ofShared();

        try {
            int[] shape = {1, 1, 1, elementCount};
            long totalBytes = (long) elementCount * Float.BYTES;
            MemorySegment aDesc = bufferTensorDesc(shape, null, totalBytes);
            MemorySegment bDesc = bufferTensorDesc(shape, null, totalBytes);
            MemorySegment yDesc = bufferTensorDesc(shape, null, totalBytes);

            // DML_ELEMENT_WISE_ADD_OPERATOR_DESC: A, B, Output (24 bytes)
            MemorySegment desc = arena.allocate(24, 8);
            desc.set(ValueLayout.ADDRESS, 0, aDesc);
            desc.set(ValueLayout.ADDRESS, 8, bDesc);
            desc.set(ValueLayout.ADDRESS, 16, yDesc);

            MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                    DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_ADD, desc);

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

            log.info("DirectMlAddKernel ready: N={}, desc={}, temp={}B, persist={}B",
                    elementCount, descriptorCount, tempSize, persistSize);
        } catch (WindowsNativeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlAddKernel" + dbg, e);
        } catch (RuntimeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlAddKernel" + dbg, e);
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

            MemorySegment outArr = arena.allocate(16, 8);
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, persistBuffer, 0, persistSize);
            outArr.set(ValueLayout.JAVA_INT, 0, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
            outArr.set(ValueLayout.ADDRESS, 8, bb);
            DirectMlBindings.bindOutputs(bt, 1, outArr);

            MemorySegment initTmp = MemorySegment.NULL;
            if (initTempSize > 0) {
                initTmp = D3D12Bindings.createDefaultBuffer(dev, initTempSize, arena);
                MemorySegment tbb = DirectMlBindings.allocBufferBinding(arena, initTmp, 0, initTempSize);
                MemorySegment tbd = DirectMlBindings.allocBindingDesc(arena,
                        DirectMlBindings.DML_BINDING_TYPE_BUFFER, tbb);
                DirectMlBindings.bindTemporaryResource(bt, tbd);
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

    public void dispatch(DirectMlTensor a, DirectMlTensor b, DirectMlTensor y)
            throws DirectMlRuntimeException {
        ensureOpen();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();
        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
            MemorySegment cl = null;
            try {
                cl = D3D12Bindings.createCommandList(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);
                recordOnto(cl, scratch, a, b, y);
                D3D12Bindings.executeOrDefer(dev, q, cl, alloc, scratch);
            } finally {
                if (cl != null) DxgiBindings.release(cl);
                DxgiBindings.release(alloc);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlAddKernel.dispatch failed" + formatDebugMessages(wb), e);
        }
    }

    /**
     * Records the elementwise-add dispatch into the caller-supplied
     * command list. See {@link DirectMlLinearKernel#recordOnto} for the
     * encoder-coalescing contract.
     */
    public void recordOnto(MemorySegment cl, Arena scratch,
                           DirectMlTensor a, DirectMlTensor b, DirectMlTensor y)
            throws DirectMlRuntimeException {
        ensureOpen();
        validate(a, "a");
        validate(b, "b");
        validate(y, "y");

        DefaultGpuBuffer ab = unwrap(a.buffer(), "a");
        DefaultGpuBuffer bb_ = unwrap(b.buffer(), "b");
        DefaultGpuBuffer yb = unwrap(y.buffer(), "y");

        MemorySegment dml = wb.getDmlDevice();
        try {
            long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, scratch);
            long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, scratch);
            MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, compiled,
                    cpuStart, gpuStart, descriptorCount);
            MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);
            try {
                MemorySegment inputs = scratch.allocate(16L * 2, 8);
                setBufferBinding(scratch, inputs, 0, ab.resource(), (long) elementCount * Float.BYTES);
                setBufferBinding(scratch, inputs, 1, bb_.resource(), (long) elementCount * Float.BYTES);
                DirectMlBindings.bindInputs(bt, 2, inputs);

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

                transitionToUav(cl, ab, scratch);
                transitionToUav(cl, bb_, scratch);
                transitionToUav(cl, yb, scratch);
                D3D12Bindings.setDescriptorHeaps(cl, descriptorHeap, scratch);
                DirectMlBindings.recordDispatch(cmdRecorder, cl, compiled, bt);
            } finally {
                DxgiBindings.release(bt);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlAddKernel.recordOnto failed" + formatDebugMessages(wb), e);
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

    private static void transitionToUav(MemorySegment cl, DefaultGpuBuffer buf, Arena scratch) {
        int state = buf.currentResourceState();
        if (state != D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12Bindings.transitionBarrier(cl, buf.resource(), state,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS, scratch);
            buf.setCurrentResourceState(D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }

    private void ensureOpen() throws DirectMlRuntimeException {
        if (closed) throw new DirectMlRuntimeException("DirectMlAddKernel already closed");
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

    public int elementCount() { return elementCount; }
}

