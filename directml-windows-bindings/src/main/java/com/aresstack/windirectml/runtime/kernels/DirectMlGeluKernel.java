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
 * Exakte GELU-Aktivierung ({@code y = 0.5·x·(1 + erf(x/√2))}) über
 * {@code DML_OPERATOR_ACTIVATION_GELU} (Enum-ID 157, eingeführt in
 * {@code DML_FEATURE_LEVEL_5_1}, verfügbar auf Windows 11 22H2+ /
 * DirectML 1.10+).
 * <p>
 * Wird von BERT-, MiniLM- und JinaBERT-Encodern in jedem
 * Feed-Forward-Block aufgerufen.
 * <p>
 * <b>Verifikation der Operator-ID:</b> Wert 157 ergibt sich durch Zählen
 * der {@code DML_OPERATOR_TYPE}-Enum-Einträge in
 * {@code %WindowsSdkDir%/Include/10.0.26100.0/um/DirectML.h}. Nach dem
 * LayerNorm-Bug (39 vs. 73) ist es Pflicht, jeden neuen Operator
 * <em>zuerst</em> gegen den offiziellen Header zu prüfen statt zu raten.
 * <p>
 * <b>Form-Konvention:</b> Der Kernel ist <em>shape-agnostisch</em>. Das
 * DML-{@code DML_BUFFER_TENSOR_DESC} wird mit der vom Caller gelieferten
 * Shape erzeugt (mindestens rank 1 wird auf rank 4 gepolstert, damit alle
 * DirectML-Treiber sicher den Path nehmen). Eingang und Ausgang müssen
 * dieselbe Element-Zahl und denselben Datentyp haben.
 * <p>
 * Operator-Desc-Layout ({@code DML_ACTIVATION_GELU_OPERATOR_DESC},
 * 16 Bytes):
 * <pre>
 * struct {
 *     const DML_TENSOR_DESC* InputTensor;   // off 0
 *     const DML_TENSOR_DESC* OutputTensor;  // off 8
 * };
 * </pre>
 * <p>
 * Lebenszyklus identisch zu {@link DirectMlLinearKernel} /
 * {@link DirectMlLayerNormKernel}: Compile + Initialize einmal im
 * Konstruktor, jede {@link #dispatch(DirectMlTensor, DirectMlTensor)}
 * baut eine frische Binding-Table und feuert eine eigene Command-List.
 */
public final class DirectMlGeluKernel implements GeluKernel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlGeluKernel.class);

    private final WindowsBindings wb;
    /**
     * Padded 4-D tensor shape used for both input and output ({@code [1, 1, 1, N]}).
     */
    private final int N;
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
     * @param ctx          initialisierter Context
     * @param elementCount Anzahl der Float-Elemente, die elementweise
     *                     transformiert werden – für den DML-Op wird die
     *                     Geometrie als {@code [1, 1, 1, elementCount]}
     *                     gewählt, was treiberunabhängig funktioniert.
     */
    public DirectMlGeluKernel(DirectMlContextImpl ctx, int elementCount)
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
        this.N = elementCount;
        this.arena = Arena.ofShared();

        try {
            // ── Tensor-Desc: [1, 1, 1, N], row-major, kein Stride-Override.
            MemorySegment xDesc = bufferTensorDesc(new int[]{1, 1, 1, N}, null,
                    (long) N * Float.BYTES);
            MemorySegment yDesc = bufferTensorDesc(new int[]{1, 1, 1, N}, null,
                    (long) N * Float.BYTES);

            // ── DML_ACTIVATION_GELU_OPERATOR_DESC (16 bytes) ──
            MemorySegment desc = arena.allocate(16, 8);
            desc.set(ValueLayout.ADDRESS, 0, xDesc);
            desc.set(ValueLayout.ADDRESS, 8, yDesc);

            MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                    DirectMlBindings.DML_OPERATOR_ACTIVATION_GELU, desc);

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

            log.info("DirectMlGeluKernel ready: N={}, desc={}, temp={}B, persist={}B",
                    N, descriptorCount, tempSize, persistSize);
        } catch (WindowsNativeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException(
                    "Failed to build DirectMlGeluKernel" + dbg, e);
        } catch (RuntimeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException(
                    "Failed to build DirectMlGeluKernel" + dbg, e);
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
                // GELU has exactly 1 input slot.
                MemorySegment inputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, inputs, 0, xb.resource(), (long) N * Float.BYTES);
                DirectMlBindings.bindInputs(bt, 1, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, yb.resource(), (long) N * Float.BYTES);
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
                    "DirectMlGeluKernel.dispatch failed" + formatDebugMessages(wb), e);
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
        if (closed) throw new DirectMlRuntimeException("DirectMlGeluKernel already closed");
    }

    private void validate(DirectMlTensor t, String name) throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32, was " + t.dataType());
        }
        if (t.shape().elementCount() != N) {
            throw new DirectMlRuntimeException(name + " must hold " + N + " elements, got "
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

    public int elementCount() {
        return N;
    }
}

