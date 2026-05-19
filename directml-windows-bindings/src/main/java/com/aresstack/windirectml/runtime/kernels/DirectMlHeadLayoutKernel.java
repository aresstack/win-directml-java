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
 * Pure data-layout kernel: reorders elements between the BERT/MiniLM "head"
 * layouts {@code [S, H, D]} and {@code [H, S, D]} (with implicit leading
 * batch dimension {@code 1}).
 * <p>
 * <b>Why this kernel exists.</b> MiniLM's Linear projections produce
 * {@code Q, K, V} as flat {@code [S, hidden] = [S, H·D]}, which is the
 * <i>"sequence-major"</i> layout {@code [S, H, D]}. The
 * {@link DirectMlAttentionKernel}, on the other hand, expects the
 * <i>"head-major"</i> layout {@code [B, H, S, D]} so that the batched
 * GEMMs treat {@code (B, H)} as the batch axes. The two layouts are
 * <b>not</b> the same memory; this kernel converts between them.
 * <p>
 * <b>How the conversion works.</b> Implemented as a single
 * {@code DML_OPERATOR_ELEMENT_WISE_IDENTITY} (op 1, FL 1.0) where the
 * <i>input</i> tensor description carries non-default strides describing
 * how to read the logical shape from the source's physical layout, and
 * the output description is plain contiguous. DirectML's identity then
 * copies element-by-element using the strided view, materialising the
 * permutation. This is the canonical, dependency-free DirectML transpose
 * pattern – it runs on every shipped {@code DirectML.dll}, including the
 * Windows-11-RTM in-box 1.8.0.
 * <p>
 * <b>Direction.</b> Two factory methods cover the two MiniLM directions:
 * <ul>
 *   <li>{@link #seqMajorToHeadMajor(DirectMlContextImpl, int, int, int)} –
 *       used before attention to split {@code [S, H·D] → [1, H, S, D]}</li>
 *   <li>{@link #headMajorToSeqMajor(DirectMlContextImpl, int, int, int)} –
 *       used after attention to merge {@code [1, H, S, D] → [S, H·D]}</li>
 * </ul>
 * Both directions use the same Identity-with-strides recipe; only the
 * stride values differ.
 * <p>
 * <b>Layout contract (precise).</b>
 * The logical shape exposed to DirectML is always 4-D
 * {@code [1, dim1, dim2, D]}; the stride array maps that logical index
 * onto the physical buffer. For batch {@code B = 1} the outer stride is
 * irrelevant – we use the total element count which is the natural value.
 * <pre>
 * forward  (seq-major → head-major): in [1, H, S, D] strides [S·H·D, D, H·D, 1] · out [1, H, S, D] contiguous
 * backward (head-major → seq-major): in [1, S, H, D] strides [S·H·D, D, S·D, 1] · out [1, S, H, D] contiguous
 * </pre>
 * The two stride vectors are exact inverses – applying both back-to-back
 * is the identity on the data (verified by the round-trip unit test).
 */
public final class DirectMlHeadLayoutKernel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlHeadLayoutKernel.class);

    /** Conversion direction enum (purely informational, used for log lines and validation). */
    public enum Direction {
        /** {@code [S, H·D]} (a.k.a. {@code [S, H, D]} contiguous) → {@code [1, H, S, D]}. */
        SEQ_TO_HEAD,
        /** {@code [1, H, S, D]} → {@code [S, H·D]}. */
        HEAD_TO_SEQ
    }

    private final WindowsBindings wb;
    private final int batch;
    private final int seq;
    private final int heads;
    private final int headDim;
    private final int elementCount;
    private final Direction direction;
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

    public static DirectMlHeadLayoutKernel seqMajorToHeadMajor(
            DirectMlContextImpl ctx, int seq, int heads, int headDim) throws DirectMlRuntimeException {
        return new DirectMlHeadLayoutKernel(ctx, 1, seq, heads, headDim, Direction.SEQ_TO_HEAD);
    }

    public static DirectMlHeadLayoutKernel headMajorToSeqMajor(
            DirectMlContextImpl ctx, int seq, int heads, int headDim) throws DirectMlRuntimeException {
        return new DirectMlHeadLayoutKernel(ctx, 1, seq, heads, headDim, Direction.HEAD_TO_SEQ);
    }

    /**
     * Batched variant – produces / consumes {@code [B, H, S, D]} tensors
     * (forward) respectively {@code [B, S, H, D]} (backward). For
     * {@code batch == 1} this is byte-identical to the legacy 4-arg
     * factory.
     */
    public static DirectMlHeadLayoutKernel seqMajorToHeadMajor(
            DirectMlContextImpl ctx, int batch, int seq, int heads, int headDim) throws DirectMlRuntimeException {
        return new DirectMlHeadLayoutKernel(ctx, batch, seq, heads, headDim, Direction.SEQ_TO_HEAD);
    }

    public static DirectMlHeadLayoutKernel headMajorToSeqMajor(
            DirectMlContextImpl ctx, int batch, int seq, int heads, int headDim) throws DirectMlRuntimeException {
        return new DirectMlHeadLayoutKernel(ctx, batch, seq, heads, headDim, Direction.HEAD_TO_SEQ);
    }

    private DirectMlHeadLayoutKernel(DirectMlContextImpl ctx, int batch, int seq, int heads, int headDim,
                                     Direction direction) throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (batch <= 0 || seq <= 0 || heads <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("batch, seq, heads, headDim must be > 0");
        }
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.batch = batch;
        this.seq = seq;
        this.heads = heads;
        this.headDim = headDim;
        this.elementCount = batch * seq * heads * headDim;
        this.direction = direction;
        this.arena = Arena.ofShared();

        try {
            final int B = batch, S = seq, H = heads, D = headDim;
            final long totalBytes = (long) elementCount * Float.BYTES;
            int[] logicalSizes;
            int[] inputStrides;

            if (direction == Direction.SEQ_TO_HEAD) {
                // Input physically [B, S, H, D]: element (b, h, s, d) at offset
                //   b·S·H·D + s·H·D + h·D + d.
                // Logical shape exposed to DML mirrors the OUTPUT layout [B, H, S, D].
                logicalSizes = new int[]{B, H, S, D};
                inputStrides = new int[]{S * H * D, D, H * D, 1};
            } else {
                // Input physically [B, H, S, D]: element (b, s, h, d) at offset
                //   b·S·H·D + h·S·D + s·D + d.
                // Logical shape exposed to DML mirrors the OUTPUT layout [B, S, H, D].
                logicalSizes = new int[]{B, S, H, D};
                inputStrides = new int[]{S * H * D, D, S * D, 1};
            }

            MemorySegment inDesc = bufferTensorDesc(logicalSizes, inputStrides, totalBytes);
            MemorySegment outDesc = bufferTensorDesc(logicalSizes, null, totalBytes);

            // DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC (24 bytes):
            //   const DML_TENSOR_DESC* InputTensor;   // off  0
            //   const DML_TENSOR_DESC* OutputTensor;  // off  8
            //   const DML_SCALE_BIAS*  ScaleBias;     // off 16 (NULL = none)
            MemorySegment desc = arena.allocate(24, 8);
            desc.set(ValueLayout.ADDRESS, 0, inDesc);
            desc.set(ValueLayout.ADDRESS, 8, outDesc);
            desc.set(ValueLayout.ADDRESS, 16, MemorySegment.NULL);

            MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                    DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_IDENTITY, desc);

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

            log.info("DirectMlHeadLayoutKernel ready: dir={}, B={}, S={}, H={}, D={}, desc={}, temp={}B, persist={}B",
                    direction, B, S, H, D, descriptorCount, tempSize, persistSize);
        } catch (WindowsNativeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException(
                    "Failed to build DirectMlHeadLayoutKernel" + dbg, e);
        } catch (RuntimeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException(
                    "Failed to build DirectMlHeadLayoutKernel" + dbg, e);
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
                    "DirectMlHeadLayoutKernel.dispatch failed" + formatDebugMessages(wb), e);
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
        if (closed) throw new DirectMlRuntimeException("DirectMlHeadLayoutKernel already closed");
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

    public Direction direction() { return direction; }

    public int batch() {
        return batch;
    }
    public int seq() { return seq; }
    public int heads() { return heads; }
    public int headDim() { return headDim; }
}

