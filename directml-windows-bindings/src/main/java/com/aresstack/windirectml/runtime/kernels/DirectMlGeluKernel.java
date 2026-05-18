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
 * Exakte GELU-Aktivierung ({@code y = 0.5·x·(1 + erf(x/√2))}) –
 * komponiert aus DirectML-FL-2.0-Primitiven, damit der Kernel auf der
 * <em>in-box</em> {@code C:\Windows\System32\DirectML.dll} jeder
 * unterstützten Windows-11-Version läuft, ohne dass eine redistributable
 * DML-DLL mit dem Projekt mitgeliefert werden muss.
 *
 * <h2>Hintergrund</h2>
 * DirectML hat seit FL 5.1 ({@code DirectML.dll} ≥ 1.10) einen nativen
 * {@code DML_OPERATOR_ACTIVATION_GELU} (Op 157, siehe
 * {@link DirectMlBindings#DML_OPERATOR_ACTIVATION_GELU}). Auf
 * Windows-11-Builds, die noch die {@code DirectML.dll 1.8.0} aus
 * Januar 2022 in {@code System32} ausliefern (z. B. Win11 21H2 LTSC),
 * existiert dieser Operator nicht – {@code CreateOperator} antwortet mit
 * {@code E_INVALIDARG}. Da die Projekt-Policy <b>keine</b> mit dem Projekt
 * mitgelieferte DirectML-DLL erlaubt (nur das vom OS gestellte System-DLL),
 * realisiert dieser Kernel GELU aus drei FL-2.0-Primitiven, die in jeder
 * versendeten DirectML-Runtime existieren:
 *
 * <pre>
 *   step 1   t1 = erf( (1/√2)·x + 0 )               // ELEMENT_WISE_ERF      (op 81)
 *   step 2   t2 = 0.5·t1 + 0.5 = 0.5·(1 + erf(...))// ELEMENT_WISE_IDENTITY (op  1)
 *   step 3   y  = x · t2                            // ELEMENT_WISE_MULTIPLY (op 24)
 * </pre>
 *
 * Drei Operatoren werden einmal compiliert; jede {@link #dispatch} baut
 * pro Sub-Op eine Binding-Table und verschachtelt die drei Aufrufe in
 * einer einzigen Command-List mit UAV-Barriers zwischen den Schritten.
 *
 * <h2>Struct-Layouts (gegen DirectML.h verifiziert, Windows SDK 26100)</h2>
 * <pre>
 * struct DML_SCALE_BIAS { FLOAT Scale; FLOAT Bias; };               // 8 Bytes
 *
 * struct DML_ELEMENT_WISE_ERF_OPERATOR_DESC {
 *     const DML_TENSOR_DESC* InputTensor;             // off  0
 *     const DML_TENSOR_DESC* OutputTensor;            // off  8
 *     const DML_SCALE_BIAS*  ScaleBias;               // off 16  (nullable)
 * }; // 24 Bytes
 *
 * struct DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC {
 *     const DML_TENSOR_DESC* InputTensor;             // off  0
 *     const DML_TENSOR_DESC* OutputTensor;            // off  8
 *     const DML_SCALE_BIAS*  ScaleBias;               // off 16  (nullable)
 * }; // 24 Bytes
 *
 * struct DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC {
 *     const DML_TENSOR_DESC* ATensor;                 // off  0
 *     const DML_TENSOR_DESC* BTensor;                 // off  8
 *     const DML_TENSOR_DESC* OutputTensor;            // off 16
 * }; // 24 Bytes
 * </pre>
 */
public final class DirectMlGeluKernel implements GeluKernel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlGeluKernel.class);

    /**
     * {@code 1/√2}, rounded to float.
     */
    private static final float INV_SQRT_2 = (float) (1.0 / Math.sqrt(2.0));

    private final WindowsBindings wb;
    private final int N;
    private final Arena arena;

    private final MemorySegment cmdRecorder;

    private final SubOp erf;
    private final SubOp identity;
    private final SubOp multiply;

    /**
     * Intermediate GPU buffers used to pass values between the three sub-ops.
     */
    private final MemorySegment t1Buf;
    private final MemorySegment t2Buf;
    private int t1State = D3D12Bindings.D3D12_RESOURCE_STATE_COMMON;
    private int t2State = D3D12Bindings.D3D12_RESOURCE_STATE_COMMON;

    private boolean closed = false;

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
            MemorySegment dml = wb.getDmlDevice();
            MemorySegment dev = wb.getD3d12Device();

            // Command recorder is shared across all three sub-operator dispatches.
            this.cmdRecorder = DirectMlBindings.createCommandRecorder(dml, arena);

            MemorySegment tDesc = bufferTensorDesc();

            // ── Step 1: ERF with ScaleBias=(1/√2, 0) ──
            MemorySegment scaleBiasErf = scaleBias(INV_SQRT_2, 0.0f);
            MemorySegment erfDesc = arena.allocate(24, 8);
            erfDesc.set(ValueLayout.ADDRESS, 0, tDesc);
            erfDesc.set(ValueLayout.ADDRESS, 8, tDesc);
            erfDesc.set(ValueLayout.ADDRESS, 16, scaleBiasErf);
            this.erf = buildSubOp(
                    DirectMlBindings.allocOperatorDesc(arena,
                            DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_ERF, erfDesc),
                    "ERF");

            // ── Step 2: IDENTITY with ScaleBias=(0.5, 0.5) ──
            MemorySegment scaleBiasIdent = scaleBias(0.5f, 0.5f);
            MemorySegment identDesc = arena.allocate(24, 8);
            identDesc.set(ValueLayout.ADDRESS, 0, tDesc);
            identDesc.set(ValueLayout.ADDRESS, 8, tDesc);
            identDesc.set(ValueLayout.ADDRESS, 16, scaleBiasIdent);
            this.identity = buildSubOp(
                    DirectMlBindings.allocOperatorDesc(arena,
                            DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_IDENTITY, identDesc),
                    "IDENTITY");

            // ── Step 3: MULTIPLY(A=x, B=t2, Output=y) ──
            MemorySegment mulDesc = arena.allocate(24, 8);
            mulDesc.set(ValueLayout.ADDRESS, 0, tDesc);
            mulDesc.set(ValueLayout.ADDRESS, 8, tDesc);
            mulDesc.set(ValueLayout.ADDRESS, 16, tDesc);
            this.multiply = buildSubOp(
                    DirectMlBindings.allocOperatorDesc(arena,
                            DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_MULTIPLY, mulDesc),
                    "MULTIPLY");

            long byteSize = (long) N * Float.BYTES;
            this.t1Buf = D3D12Bindings.createDefaultBuffer(dev, byteSize, arena);
            this.t2Buf = D3D12Bindings.createDefaultBuffer(dev, byteSize, arena);

            log.info("DirectMlGeluKernel (composite ERF+IDENTITY+MULTIPLY) ready: N={}, " +
                            "desc(erf/id/mul)={}/{}/{}, " +
                            "temp(erf/id/mul)={}/{}/{}B, " +
                            "persist(erf/id/mul)={}/{}/{}B",
                    N,
                    erf.descCount, identity.descCount, multiply.descCount,
                    erf.tempSize, identity.tempSize, multiply.tempSize,
                    erf.persistSize, identity.persistSize, multiply.persistSize);
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

    /**
     * Compile + (optionally) initialize one sub-operator.
     */
    private SubOp buildSubOp(MemorySegment opDesc, String name) throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        MemorySegment compiled = DirectMlBindings.compileOperator(dml, op,
                DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);

        long[] bp = DirectMlBindings.getBindingProperties(compiled, arena);
        int descCount = Math.max((int) bp[0], 1);
        long tempSize = bp[1];
        long persistSize = bp[2];

        MemorySegment heap = D3D12Bindings.createDescriptorHeap(dev, descCount, arena);
        MemorySegment tempBuf = tempSize > 0
                ? D3D12Bindings.createDefaultBuffer(dev, tempSize, arena) : MemorySegment.NULL;
        MemorySegment persistBuf = persistSize > 0
                ? D3D12Bindings.createDefaultBuffer(dev, persistSize, arena) : MemorySegment.NULL;

        SubOp sub = new SubOp(name, compiled, heap, descCount,
                tempSize, persistSize, tempBuf, persistBuf);
        if (persistSize > 0) {
            initializeSubOp(sub);
        }
        return sub;
    }

    private void initializeSubOp(SubOp sub) throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();

        MemorySegment initializer = DirectMlBindings.createOperatorInitializer(dml,
                new MemorySegment[]{sub.compiled}, arena);
        long[] ibp = DirectMlBindings.getBindingProperties(initializer, arena);
        int initDescCount = Math.max((int) ibp[0], 1);
        long initTempSize = ibp[1];

        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(sub.heap, arena);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(sub.heap, arena);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, initializer,
                cpuStart, gpuStart, initDescCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);
        try {
            MemorySegment inputBindings = DirectMlBindings.allocNoneBindingDesc(arena);
            DirectMlBindings.bindInputs(bt, 1, inputBindings);

            if (sub.persistSize > 0) {
                MemorySegment outArr = arena.allocate(16, 8);
                MemorySegment bb = DirectMlBindings.allocBufferBinding(arena,
                        sub.persistBuf, 0, sub.persistSize);
                outArr.set(ValueLayout.JAVA_INT, 0, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
                outArr.set(ValueLayout.ADDRESS, 8, bb);
                DirectMlBindings.bindOutputs(bt, 1, outArr);
            }

            if (initTempSize > 0) {
                MemorySegment initTmp = D3D12Bindings.createDefaultBuffer(dev, initTempSize, arena);
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
                D3D12Bindings.setDescriptorHeaps(cl, sub.heap, arena);
                DirectMlBindings.recordDispatch(cmdRecorder, cl, initializer, bt);
                D3D12Bindings.executeAndWait(dev, q, cl, arena);
            } finally {
                if (cl != null) DxgiBindings.release(cl);
                DxgiBindings.release(alloc);
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

        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();
        long byteSize = (long) N * Float.BYTES;

        try (Arena scratch = Arena.ofConfined()) {
            // Per-sub-op binding tables (one descriptor heap per sub-op).
            MemorySegment btErf = buildOneInOneOut(scratch, erf,
                    xb.resource(), t1Buf, byteSize);
            MemorySegment btId = buildOneInOneOut(scratch, identity,
                    t1Buf, t2Buf, byteSize);
            MemorySegment btMul = buildTwoInOneOut(scratch, multiply,
                    xb.resource(), t2Buf, yb.resource(), byteSize);

            MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
            MemorySegment cl = null;
            try {
                cl = D3D12Bindings.createCommandList(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);

                // 1) Transition all bound buffers to UAV.
                transitionToUav(cl, xb, scratch);
                transitionToUav(cl, yb, scratch);
                t1State = ensureUav(cl, t1Buf, t1State, scratch);
                t2State = ensureUav(cl, t2Buf, t2State, scratch);

                // 2) Step 1: ERF — x → t1
                D3D12Bindings.setDescriptorHeaps(cl, erf.heap, scratch);
                DirectMlBindings.recordDispatch(cmdRecorder, cl, erf.compiled, btErf);
                D3D12Bindings.uavBarrier(cl, scratch);

                // 3) Step 2: IDENTITY — t1 → t2  (applies 0.5·t1 + 0.5)
                D3D12Bindings.setDescriptorHeaps(cl, identity.heap, scratch);
                DirectMlBindings.recordDispatch(cmdRecorder, cl, identity.compiled, btId);
                D3D12Bindings.uavBarrier(cl, scratch);

                // 4) Step 3: MULTIPLY — x · t2 → y
                D3D12Bindings.setDescriptorHeaps(cl, multiply.heap, scratch);
                DirectMlBindings.recordDispatch(cmdRecorder, cl, multiply.compiled, btMul);

                D3D12Bindings.executeAndWait(dev, q, cl, scratch);
            } finally {
                if (cl != null) DxgiBindings.release(cl);
                DxgiBindings.release(alloc);
                DxgiBindings.release(btErf);
                DxgiBindings.release(btId);
                DxgiBindings.release(btMul);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlGeluKernel.dispatch failed" + formatDebugMessages(wb), e);
        }
    }

    /**
     * Build a binding table for an ERF / IDENTITY style op (1 input, 1 output).
     */
    private MemorySegment buildOneInOneOut(Arena scratch, SubOp sub,
                                           MemorySegment inputRes, MemorySegment outputRes,
                                           long byteSize) throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(sub.heap, scratch);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(sub.heap, scratch);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, sub.compiled,
                cpuStart, gpuStart, sub.descCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);

        MemorySegment inputs = scratch.allocate(16, 8);
        setBufferBinding(scratch, inputs, 0, inputRes, byteSize);
        DirectMlBindings.bindInputs(bt, 1, inputs);

        MemorySegment outputs = scratch.allocate(16, 8);
        setBufferBinding(scratch, outputs, 0, outputRes, byteSize);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(scratch, bt, sub);
        return bt;
    }

    /**
     * Build a binding table for MULTIPLY (2 inputs, 1 output).
     */
    private MemorySegment buildTwoInOneOut(Arena scratch, SubOp sub,
                                           MemorySegment aRes, MemorySegment bRes,
                                           MemorySegment outRes, long byteSize)
            throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(sub.heap, scratch);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(sub.heap, scratch);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, sub.compiled,
                cpuStart, gpuStart, sub.descCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);

        MemorySegment inputs = scratch.allocate(16L * 2, 8);
        setBufferBinding(scratch, inputs, 0, aRes, byteSize);
        setBufferBinding(scratch, inputs, 1, bRes, byteSize);
        DirectMlBindings.bindInputs(bt, 2, inputs);

        MemorySegment outputs = scratch.allocate(16, 8);
        setBufferBinding(scratch, outputs, 0, outRes, byteSize);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(scratch, bt, sub);
        return bt;
    }

    private void bindTempAndPersist(Arena scratch, MemorySegment bt, SubOp sub) {
        if (sub.tempSize > 0) {
            MemorySegment binding = DirectMlBindings.allocBufferBinding(scratch,
                    sub.tempBuf, 0, sub.tempSize);
            DirectMlBindings.bindTemporaryResource(bt,
                    DirectMlBindings.allocBindingDesc(scratch,
                            DirectMlBindings.DML_BINDING_TYPE_BUFFER, binding));
        }
        if (sub.persistSize > 0) {
            MemorySegment binding = DirectMlBindings.allocBufferBinding(scratch,
                    sub.persistBuf, 0, sub.persistSize);
            DirectMlBindings.bindPersistentResource(bt,
                    DirectMlBindings.allocBindingDesc(scratch,
                            DirectMlBindings.DML_BINDING_TYPE_BUFFER, binding));
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        for (SubOp sub : new SubOp[]{multiply, identity, erf}) {
            if (sub == null) continue;
            try {
                DxgiBindings.release(sub.heap);
            } catch (Exception ignored) {
            }
            try {
                DxgiBindings.release(sub.compiled);
            } catch (Exception ignored) {
            }
            if (!sub.persistBuf.equals(MemorySegment.NULL)) {
                try {
                    DxgiBindings.release(sub.persistBuf);
                } catch (Exception ignored) {
                }
            }
            if (!sub.tempBuf.equals(MemorySegment.NULL)) {
                try {
                    DxgiBindings.release(sub.tempBuf);
                } catch (Exception ignored) {
                }
            }
        }
        if (t2Buf != null) {
            try {
                DxgiBindings.release(t2Buf);
            } catch (Exception ignored) {
            }
        }
        if (t1Buf != null) {
            try {
                DxgiBindings.release(t1Buf);
            } catch (Exception ignored) {
            }
        }
        try {
            DxgiBindings.release(cmdRecorder); } catch (Exception ignored) { }
        arena.close();
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private static String formatDebugMessages(WindowsBindings wb) {
        java.util.List<String> msgs = wb.drainDebugMessages();
        if (msgs.isEmpty()) return "";
        StringBuilder sb = new StringBuilder("\n--- DirectML/D3D12 debug messages ---");
        for (String m : msgs) sb.append("\n  ").append(m);
        return sb.toString();
    }

    private static void transitionToUav(MemorySegment cl, DefaultGpuBuffer buf, Arena scratch) {
        int state = buf.currentResourceState();
        if (state != D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12Bindings.transitionBarrier(cl, buf.resource(), state,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS, scratch);
            buf.setCurrentResourceState(D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }

    private static int ensureUav(MemorySegment cl, MemorySegment res, int currentState, Arena scratch) {
        int uav = D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        if (currentState != uav) {
            D3D12Bindings.transitionBarrier(cl, res, currentState, uav, scratch);
        }
        return uav;
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

    private MemorySegment bufferTensorDesc() {
        MemorySegment buf = DirectMlBindings.allocBufferTensorDesc(arena,
                DirectMlBindings.DML_TENSOR_DATA_TYPE_FLOAT32,
                new int[]{1, 1, 1, N}, null, (long) N * Float.BYTES);
        return DirectMlBindings.allocTensorDesc(arena, buf);
    }

    /**
     * Allocate a {@code DML_SCALE_BIAS} struct (8 bytes) in the kernel arena.
     */
    private MemorySegment scaleBias(float scale, float bias) {
        MemorySegment sb = arena.allocate(8, 4);
        sb.set(ValueLayout.JAVA_FLOAT, 0, scale);
        sb.set(ValueLayout.JAVA_FLOAT, 4, bias);
        return sb;
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

    // ── one record per sub-operator ─────────────────────────────────────

    private static final class SubOp {
        final String name;
        final MemorySegment compiled;
        final MemorySegment heap;
        final int descCount;
        final long tempSize;
        final long persistSize;
        final MemorySegment tempBuf;     // may be NULL
        final MemorySegment persistBuf;  // may be NULL

        SubOp(String name, MemorySegment compiled, MemorySegment heap, int descCount,
              long tempSize, long persistSize,
              MemorySegment tempBuf, MemorySegment persistBuf) {
            this.name = name;
            this.compiled = compiled;
            this.heap = heap;
            this.descCount = descCount;
            this.tempSize = tempSize;
            this.persistSize = persistSize;
            this.tempBuf = tempBuf;
            this.persistBuf = persistBuf;
        }
    }
}

