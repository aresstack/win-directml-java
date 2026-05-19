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
 * Composite implementation of exact GELU ({@code y = 0.5·x·(1 + erf(x/√2))})
 * built from three FL-2.0 primitives:
 * <ol>
 *   <li>{@code DML_OPERATOR_ELEMENT_WISE_ERF} (op 81, FL 2.0) with built-in
 *       {@code DML_SCALE_BIAS(scale=1/√2, bias=0)} →
 *       {@code tmpA = erf(x / √2)}.</li>
 *   <li>{@code DML_OPERATOR_ELEMENT_WISE_IDENTITY} (op 1, FL 1.0) with
 *       {@code DML_SCALE_BIAS(scale=0.5, bias=0.5)} →
 *       {@code tmpB = 0.5·tmpA + 0.5 = 0.5·(1 + erf(x/√2))}.</li>
 *   <li>{@code DML_OPERATOR_ELEMENT_WISE_MULTIPLY} (op 24, FL 1.0) →
 *       {@code y = x · tmpB}.</li>
 * </ol>
 * This is the fallback for hosts whose in-box {@code DirectML.dll} does
 * not expose {@link DirectMlBindings#DML_OPERATOR_ACTIVATION_GELU}
 * (introduced in {@code DML_FEATURE_LEVEL_5_1}, Windows 11 22H2 /
 * DirectML 1.10). All three primitive ops are present in every shipping
 * {@code DirectML.dll} on Windows 10/11 since the original release.
 * <p>
 * <b>Lifecycle:</b> Each sub-op owns its own compiled operator, descriptor
 * heap, optional temp/persistent buffers and is initialised in the
 * constructor. The kernel additionally owns two intermediate
 * {@code N}-float UAV buffers ({@code tmpA}, {@code tmpB}) that hold the
 * sub-step results across the three dispatches.
 * <p>
 * <b>Dispatch cost:</b> Each {@link #dispatch} call submits three command
 * lists with {@code executeAndWait} between them (each sub-op uses its
 * own descriptor heap to avoid cross-op aliasing). On a 6-layer MiniLM
 * encoder this adds 12 extra GPU fences per inference compared to the
 * single-fused-GELU path; folding all three into one command list is a
 * follow-up optimisation that is not required for correctness.
 */
public final class DirectMlCompositeGeluKernel implements GeluKernel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlCompositeGeluKernel.class);

    private static final float INV_SQRT_2 = (float) (1.0 / Math.sqrt(2.0));

    private final WindowsBindings wb;
    private final int N;
    private final Arena arena;

    private final MiniOp erf;
    private final MiniOp identity;
    private final MiniOp multiply;
    private final MemorySegment cmdRecorder;
    private final MemorySegment tmpA;          // intermediate UAV buffer, N floats
    private final MemorySegment tmpB;          // intermediate UAV buffer, N floats

    // Tracked resource states so we can emit the right transition barriers
    // for the two intermediate buffers.
    private int tmpAState = D3D12Bindings.D3D12_RESOURCE_STATE_COMMON;
    private int tmpBState = D3D12Bindings.D3D12_RESOURCE_STATE_COMMON;

    private boolean closed = false;

    /**
     * Bundle: one compiled DirectML operator + its descriptor heap and
     * temp/persistent buffers, ready for {@code RecordDispatch}.
     */
    private static final class MiniOp {
        final MemorySegment compiled;
        final MemorySegment descriptorHeap;
        final int descriptorCount;
        final long tempSize;
        final long persistSize;
        final MemorySegment tempBuffer;
        final MemorySegment persistBuffer;

        MiniOp(MemorySegment compiled, MemorySegment descriptorHeap, int descriptorCount,
               long tempSize, long persistSize,
               MemorySegment tempBuffer, MemorySegment persistBuffer) {
            this.compiled = compiled;
            this.descriptorHeap = descriptorHeap;
            this.descriptorCount = descriptorCount;
            this.tempSize = tempSize;
            this.persistSize = persistSize;
            this.tempBuffer = tempBuffer;
            this.persistBuffer = persistBuffer;
        }
    }

    public DirectMlCompositeGeluKernel(DirectMlContextImpl ctx, int elementCount)
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

        MiniOp erfOp = null, identityOp = null, mulOp = null;
        MemorySegment recorder = MemorySegment.NULL;
        MemorySegment tA = MemorySegment.NULL, tB = MemorySegment.NULL;
        try {
            MemorySegment dml = wb.getDmlDevice();
            MemorySegment dev = wb.getD3d12Device();
            recorder = DirectMlBindings.createCommandRecorder(dml, arena);

            // ── sub-op 1: ERF with ScaleBias(1/√2, 0) ───────────────────
            erfOp = buildUnaryWithScaleBias(dml, dev,
                    DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_ERF,
                    INV_SQRT_2, 0f);

            // ── sub-op 2: IDENTITY with ScaleBias(0.5, 0.5) ─────────────
            identityOp = buildUnaryWithScaleBias(dml, dev,
                    DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                    0.5f, 0.5f);

            // ── sub-op 3: MULTIPLY (no ScaleBias) ───────────────────────
            mulOp = buildBinaryMultiply(dml, dev);

            // ── intermediate buffers ────────────────────────────────────
            long bytes = (long) N * Float.BYTES;
            tA = D3D12Bindings.createDefaultBuffer(dev, bytes, arena);
            tB = D3D12Bindings.createDefaultBuffer(dev, bytes, arena);

            this.erf = erfOp;
            this.identity = identityOp;
            this.multiply = mulOp;
            this.cmdRecorder = recorder;
            this.tmpA = tA;
            this.tmpB = tB;

            log.info("DirectMlCompositeGeluKernel ready: N={}  (ERF+IDENTITY+MUL composite, FL-2.0 path)", N);
        } catch (WindowsNativeException | RuntimeException e) {
            String dbg = formatDebugMessages(wb);
            // Best-effort cleanup of anything we partially built.
            if (mulOp != null) closeMiniOp(mulOp);
            if (identityOp != null) closeMiniOp(identityOp);
            if (erfOp != null) closeMiniOp(erfOp);
            if (!tB.equals(MemorySegment.NULL)) try {
                DxgiBindings.release(tB);
            } catch (Exception ignored) {
            }
            if (!tA.equals(MemorySegment.NULL)) try {
                DxgiBindings.release(tA);
            } catch (Exception ignored) {
            }
            if (!recorder.equals(MemorySegment.NULL)) try {
                DxgiBindings.release(recorder);
            } catch (Exception ignored) {
            }
            arena.close();
            throw new DirectMlRuntimeException(
                    "Failed to build DirectMlCompositeGeluKernel" + dbg, e);
        }
    }

    /**
     * Build either ERF (op 81) or IDENTITY (op 1) – both share the
     * same {@code (input, output, scaleBias)} 24-byte layout.
     */
    private MiniOp buildUnaryWithScaleBias(MemorySegment dml, MemorySegment dev,
                                           int opType, float scale, float bias)
            throws WindowsNativeException {
        int[] shape = {1, 1, 1, N};
        long bytes = (long) N * Float.BYTES;
        MemorySegment inDesc = bufferTensorDesc(shape, bytes);
        MemorySegment outDesc = bufferTensorDesc(shape, bytes);

        // DML_SCALE_BIAS: { FLOAT Scale; FLOAT Bias; } (8 bytes)
        MemorySegment scaleBias = arena.allocate(8, 4);
        scaleBias.set(ValueLayout.JAVA_FLOAT, 0, scale);
        scaleBias.set(ValueLayout.JAVA_FLOAT, 4, bias);

        // DML_ELEMENT_WISE_{ERF,IDENTITY}_OPERATOR_DESC (24 bytes)
        MemorySegment desc = arena.allocate(24, 8);
        desc.set(ValueLayout.ADDRESS, 0, inDesc);
        desc.set(ValueLayout.ADDRESS, 8, outDesc);
        desc.set(ValueLayout.ADDRESS, 16, scaleBias);

        return compileAndInit(dml, dev, opType, desc, /* persistOutputs */ true);
    }

    private MiniOp buildBinaryMultiply(MemorySegment dml, MemorySegment dev)
            throws WindowsNativeException {
        int[] shape = {1, 1, 1, N};
        long bytes = (long) N * Float.BYTES;
        MemorySegment aDesc = bufferTensorDesc(shape, bytes);
        MemorySegment bDesc = bufferTensorDesc(shape, bytes);
        MemorySegment oDesc = bufferTensorDesc(shape, bytes);

        // DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC (24 bytes; no ScaleBias slot)
        MemorySegment desc = arena.allocate(24, 8);
        desc.set(ValueLayout.ADDRESS, 0, aDesc);
        desc.set(ValueLayout.ADDRESS, 8, bDesc);
        desc.set(ValueLayout.ADDRESS, 16, oDesc);

        return compileAndInit(dml, dev,
                DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                desc, /* persistOutputs */ true);
    }

    private MiniOp compileAndInit(MemorySegment dml, MemorySegment dev,
                                  int opType, MemorySegment innerDesc,
                                  boolean persistOutputs) throws WindowsNativeException {
        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena, opType, innerDesc);
        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        MemorySegment compiled = DirectMlBindings.compileOperator(dml, op,
                DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);

        long[] bp = DirectMlBindings.getBindingProperties(compiled, arena);
        int descCount = Math.max((int) bp[0], 1);
        long tempSize = bp[1];
        long persistSize = bp[2];

        MemorySegment heap = D3D12Bindings.createDescriptorHeap(dev, descCount, arena);
        MemorySegment temp = (tempSize > 0)
                ? D3D12Bindings.createDefaultBuffer(dev, tempSize, arena) : MemorySegment.NULL;
        MemorySegment persist = (persistSize > 0)
                ? D3D12Bindings.createDefaultBuffer(dev, persistSize, arena) : MemorySegment.NULL;

        MiniOp m = new MiniOp(compiled, heap, descCount, tempSize, persistSize, temp, persist);
        if (persistSize > 0 && persistOutputs) {
            initializeOperator(m);
        }
        return m;
    }

    private void initializeOperator(MiniOp m) throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();

        MemorySegment initializer = DirectMlBindings.createOperatorInitializer(dml,
                new MemorySegment[]{m.compiled}, arena);
        long[] ibp = DirectMlBindings.getBindingProperties(initializer, arena);
        int initDescCount = Math.max((int) ibp[0], 1);
        long initTempSize = ibp[1];

        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(m.descriptorHeap, arena);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(m.descriptorHeap, arena);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, initializer,
                cpuStart, gpuStart, initDescCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);
        try {
            MemorySegment inputBindings = DirectMlBindings.allocNoneBindingDesc(arena);
            DirectMlBindings.bindInputs(bt, 1, inputBindings);

            if (m.persistSize > 0) {
                MemorySegment outArr = arena.allocate(16, 8);
                MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, m.persistBuffer, 0, m.persistSize);
                outArr.set(ValueLayout.JAVA_INT, 0, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
                outArr.set(ValueLayout.ADDRESS, 8, bb);
                DirectMlBindings.bindOutputs(bt, 1, outArr);
            }

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
                D3D12Bindings.setDescriptorHeaps(cl, m.descriptorHeap, arena);
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

    // ─────────────────────────────────────────────────────────────────
    // Dispatch
    // ─────────────────────────────────────────────────────────────────

    @Override
    public void dispatch(DirectMlTensor x, DirectMlTensor y) throws DirectMlRuntimeException {
        ensureOpen();
        validate(x, "x");
        validate(y, "y");

        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();
        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
            MemorySegment cl = null;
            try {
                cl = D3D12Bindings.createCommandList(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);
                recordOnto(cl, scratch, x, y);
                D3D12Bindings.executeOrDefer(dev, q, cl, alloc, scratch);
            } finally {
                if (cl != null) DxgiBindings.release(cl);
                DxgiBindings.release(alloc);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlCompositeGeluKernel.dispatch failed" + formatDebugMessages(wb), e);
        }
    }

    @Override
    public void recordOnto(MemorySegment cl, Arena scratch,
                           DirectMlTensor x, DirectMlTensor y) throws DirectMlRuntimeException {
        ensureOpen();
        validate(x, "x");
        validate(y, "y");

        DefaultGpuBuffer xb = unwrap(x.buffer(), "x");
        DefaultGpuBuffer yb = unwrap(y.buffer(), "y");

        try {
            // 1. tmpA = erf(x / √2)            (uses ScaleBias on input)
            recordUnary(cl, scratch, erf, xb.resource(), xb, tmpA, /* outIsInternal */ true,
                    () -> tmpAState, s -> tmpAState = s,
                    /* inStateAccessor */ null, /* inStateSetter */ null);
            D3D12Bindings.uavBarrier(cl, scratch);

            // 2. tmpB = 0.5*tmpA + 0.5         (ScaleBias on input again)
            recordUnary(cl, scratch, identity, tmpA, /* inputBuf */ null, tmpB, /* outIsInternal */ true,
                    () -> tmpBState, s -> tmpBState = s,
                    /* inStateAccessor */ () -> tmpAState,
                    /* inStateSetter   */ s -> tmpAState = s);
            D3D12Bindings.uavBarrier(cl, scratch);

            // 3. y = x * tmpB                   (binary)
            recordBinaryMul(cl, scratch, xb, tmpB, yb);
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlCompositeGeluKernel.recordOnto failed" + formatDebugMessages(wb), e);
        }
    }

    /**
     * Records a unary op into the caller-supplied command list. Does
     * <b>not</b> create allocator/list and does <b>not</b> submit/wait.
     */
    private void recordUnary(MemorySegment cl, Arena scratch, MiniOp op,
                             MemorySegment inputRes, DefaultGpuBuffer inputBuf,
                             MemorySegment outputRes, boolean outIsInternal,
                             java.util.function.IntSupplier getOutState,
                             java.util.function.IntConsumer setOutState,
                             java.util.function.IntSupplier getInState,
                             java.util.function.IntConsumer setInState) throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(op.descriptorHeap, scratch);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(op.descriptorHeap, scratch);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, op.compiled,
                cpuStart, gpuStart, op.descriptorCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);
        try {
            MemorySegment inputs = scratch.allocate(16, 8);
            setBufferBinding(scratch, inputs, 0, inputRes, (long) N * Float.BYTES);
            DirectMlBindings.bindInputs(bt, 1, inputs);

            MemorySegment outputs = scratch.allocate(16, 8);
            setBufferBinding(scratch, outputs, 0, outputRes, (long) N * Float.BYTES);
            DirectMlBindings.bindOutputs(bt, 1, outputs);

            bindTempPersist(scratch, bt, op);

            if (inputBuf != null) {
                transitionToUav(cl, inputBuf, scratch);
            } else if (getInState != null) {
                transitionInternalToUav(cl, inputRes, getInState, setInState, scratch);
            }
            transitionInternalToUav(cl, outputRes, getOutState, setOutState, scratch);
            D3D12Bindings.setDescriptorHeaps(cl, op.descriptorHeap, scratch);
            DirectMlBindings.recordDispatch(cmdRecorder, cl, op.compiled, bt);
        } finally {
            DxgiBindings.release(bt);
        }
    }

    /**
     * Records the final {@code y = x * tmpB} multiply into the
     * caller-supplied command list.
     */
    private void recordBinaryMul(MemorySegment cl, Arena scratch,
                                 DefaultGpuBuffer xb, MemorySegment tmpBRes, DefaultGpuBuffer yb)
            throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(multiply.descriptorHeap, scratch);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(multiply.descriptorHeap, scratch);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, multiply.compiled,
                cpuStart, gpuStart, multiply.descriptorCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);
        try {
            MemorySegment inputs = scratch.allocate(16L * 2, 8);
            setBufferBinding(scratch, inputs, 0, xb.resource(), (long) N * Float.BYTES);
            setBufferBinding(scratch, inputs, 1, tmpBRes, (long) N * Float.BYTES);
            DirectMlBindings.bindInputs(bt, 2, inputs);

            MemorySegment outputs = scratch.allocate(16, 8);
            setBufferBinding(scratch, outputs, 0, yb.resource(), (long) N * Float.BYTES);
            DirectMlBindings.bindOutputs(bt, 1, outputs);

            bindTempPersist(scratch, bt, multiply);

            transitionToUav(cl, xb, scratch);
            transitionInternalToUav(cl, tmpBRes, () -> tmpBState, s -> tmpBState = s, scratch);
            transitionToUav(cl, yb, scratch);
            D3D12Bindings.setDescriptorHeaps(cl, multiply.descriptorHeap, scratch);
            DirectMlBindings.recordDispatch(cmdRecorder, cl, multiply.compiled, bt);
        } finally {
            DxgiBindings.release(bt);
        }
    }

    private void bindTempPersist(Arena scratch, MemorySegment bt, MiniOp op) {
        if (op.tempSize > 0) {
            MemorySegment binding = DirectMlBindings.allocBufferBinding(scratch,
                    op.tempBuffer, 0, op.tempSize);
            DirectMlBindings.bindTemporaryResource(bt,
                    DirectMlBindings.allocBindingDesc(scratch,
                            DirectMlBindings.DML_BINDING_TYPE_BUFFER, binding));
        }
        if (op.persistSize > 0) {
            MemorySegment binding = DirectMlBindings.allocBufferBinding(scratch,
                    op.persistBuffer, 0, op.persistSize);
            DirectMlBindings.bindPersistentResource(bt,
                    DirectMlBindings.allocBindingDesc(scratch,
                            DirectMlBindings.DML_BINDING_TYPE_BUFFER, binding));
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
        closeMiniOp(multiply);
        closeMiniOp(identity);
        closeMiniOp(erf);
        if (!tmpB.equals(MemorySegment.NULL)) try {
            DxgiBindings.release(tmpB);
        } catch (Exception ignored) {
        }
        if (!tmpA.equals(MemorySegment.NULL)) try {
            DxgiBindings.release(tmpA);
        } catch (Exception ignored) {
        }
        arena.close();
    }

    private static void closeMiniOp(MiniOp m) {
        try {
            DxgiBindings.release(m.descriptorHeap);
        } catch (Exception ignored) {
        }
        try {
            DxgiBindings.release(m.compiled);
        } catch (Exception ignored) {
        }
        if (!m.persistBuffer.equals(MemorySegment.NULL)) {
            try {
                DxgiBindings.release(m.persistBuffer);
            } catch (Exception ignored) {
            }
        }
        if (!m.tempBuffer.equals(MemorySegment.NULL)) {
            try {
                DxgiBindings.release(m.tempBuffer);
            } catch (Exception ignored) {
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // helpers
    // ─────────────────────────────────────────────────────────────────

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

    private static void transitionInternalToUav(MemorySegment cl, MemorySegment res,
                                                java.util.function.IntSupplier getState,
                                                java.util.function.IntConsumer setState,
                                                Arena scratch) {
        int state = getState.getAsInt();
        if (state != D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12Bindings.transitionBarrier(cl, res, state,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS, scratch);
            setState.accept(D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }

    private void ensureOpen() throws DirectMlRuntimeException {
        if (closed) throw new DirectMlRuntimeException("DirectMlCompositeGeluKernel already closed");
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

    public int elementCount() {
        return N;
    }
}

