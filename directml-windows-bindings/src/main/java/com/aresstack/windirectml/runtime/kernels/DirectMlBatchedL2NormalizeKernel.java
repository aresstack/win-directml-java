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
 * Batched DirectML L2 normalisation: {@code N} independent
 * {@code [H]} → {@code [H]} unit-norm operations in one kernel instance.
 * <p>
 * Pipeline (FL-1.0 only):
 * <ol>
 *   <li>Per-row sum-of-squares via batched GEMM
 *       {@code A[N,1,1,H] · B[N,1,H,1] = S[N,1,1,1]} (A=B=x;
 *       both transforms NONE, the column shape comes from the B
 *       descriptor). {@code N} dot products in one dispatch.</li>
 *   <li>Element-wise SQRT with {@code DML_SCALE_BIAS(scale=1, bias=ε²)}
 *       on the {@code N}-element scalar tensor → norm per row.</li>
 *   <li>Broadcast divide
 *       {@code y[N,1,1,H] = x[N,1,1,H] / n[N,1,1,H]} where {@code n}
 *       has strides {@code [1,0,0,0]} so each of the {@code H} lanes
 *       in row {@code k} reads the {@code k}-th norm float.</li>
 * </ol>
 * The constructor compiles + initialises the three sub-ops for a fixed
 * {@code (N, H, ε)} triple. Recompiling for a different {@code ε}
 * defeats the bake-into-bias optimisation, so {@link #dispatch} fails
 * loudly if a caller passes a mismatched epsilon.
 */
public final class DirectMlBatchedL2NormalizeKernel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlBatchedL2NormalizeKernel.class);

    private final WindowsBindings wb;
    private final int N;
    private final int H;
    private final float epsilon;
    private final Arena arena;

    private final MiniOp sumSqOp;
    private final MiniOp sqrtOp;
    private final MiniOp divideOp;
    private final MemorySegment cmdRecorder;

    private final MemorySegment sBuf;   // [N] floats – sum of squares
    private final MemorySegment nBuf;   // [N] floats – norms
    private int sBufState = D3D12Bindings.D3D12_RESOURCE_STATE_COMMON;
    private int nBufState = D3D12Bindings.D3D12_RESOURCE_STATE_COMMON;

    private boolean closed = false;

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

    public DirectMlBatchedL2NormalizeKernel(DirectMlContextImpl ctx, int batch,
                                            int elementCount, float epsilon)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (batch <= 0 || elementCount <= 0) {
            throw new IllegalArgumentException("batch, elementCount must be > 0");
        }
        if (!(epsilon >= 0f) || Float.isNaN(epsilon) || Float.isInfinite(epsilon)) {
            throw new IllegalArgumentException("epsilon must be finite and >= 0, was " + epsilon);
        }
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.N = batch;
        this.H = elementCount;
        this.epsilon = epsilon;
        this.arena = Arena.ofShared();

        MiniOp gemmOp = null, sqrtMini = null, divMini = null;
        MemorySegment recorder = MemorySegment.NULL;
        MemorySegment s = MemorySegment.NULL, n = MemorySegment.NULL;
        try {
            MemorySegment dml = wb.getDmlDevice();
            MemorySegment dev = wb.getD3d12Device();
            recorder = DirectMlBindings.createCommandRecorder(dml, arena);

            gemmOp = buildBatchedSumOfSquaresGemm(dml, dev);
            float epsSq = epsilon * epsilon;
            sqrtMini = buildBatchedSqrt(dml, dev, 1.0f, epsSq);
            divMini = buildBatchedBroadcastDivide(dml, dev);

            s = D3D12Bindings.createDefaultBuffer(dev, (long) N * Float.BYTES, arena);
            n = D3D12Bindings.createDefaultBuffer(dev, (long) N * Float.BYTES, arena);

            this.sumSqOp = gemmOp;
            this.sqrtOp = sqrtMini;
            this.divideOp = divMini;
            this.cmdRecorder = recorder;
            this.sBuf = s;
            this.nBuf = n;

            log.info("DirectMlBatchedL2NormalizeKernel ready: N={}, H={}, ε={} (batched GEMM+SQRT+DIVIDE, FL-1.0)",
                    N, H, epsilon);
        } catch (WindowsNativeException | RuntimeException e) {
            String dbg = formatDebugMessages(wb);
            if (divMini != null) closeMiniOp(divMini);
            if (sqrtMini != null) closeMiniOp(sqrtMini);
            if (gemmOp != null) closeMiniOp(gemmOp);
            if (!n.equals(MemorySegment.NULL)) try {
                DxgiBindings.release(n);
            } catch (Exception ignored) {
            }
            if (!s.equals(MemorySegment.NULL)) try {
                DxgiBindings.release(s);
            } catch (Exception ignored) {
            }
            if (!recorder.equals(MemorySegment.NULL)) try {
                DxgiBindings.release(recorder);
            } catch (Exception ignored) {
            }
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlBatchedL2NormalizeKernel" + dbg, e);
        }
    }

    // ── sub-op builders ─────────────────────────────────────────────────

    /**
     * A[N,1,1,H] · B[N,1,H,1] = S[N,1,1,1]; A and B alias the same buffer.
     */
    private MiniOp buildBatchedSumOfSquaresGemm(MemorySegment dml, MemorySegment dev)
            throws WindowsNativeException {
        long xBytes = (long) N * H * Float.BYTES;
        MemorySegment aDesc = bufferTensorDesc(new int[]{N, 1, 1, H}, null, xBytes);
        MemorySegment bDesc = bufferTensorDesc(new int[]{N, 1, H, 1}, null, xBytes);
        MemorySegment yDesc = bufferTensorDesc(new int[]{N, 1, 1, 1}, null, (long) N * Float.BYTES);

        MemorySegment gemm = arena.allocate(56, 8);
        gemm.set(ValueLayout.ADDRESS, 0, aDesc);
        gemm.set(ValueLayout.ADDRESS, 8, bDesc);
        gemm.set(ValueLayout.ADDRESS, 16, MemorySegment.NULL);
        gemm.set(ValueLayout.ADDRESS, 24, yDesc);
        gemm.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
        gemm.set(ValueLayout.JAVA_INT, 36, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
        gemm.set(ValueLayout.JAVA_FLOAT, 40, 1.0f);
        gemm.set(ValueLayout.JAVA_FLOAT, 44, 0.0f);
        gemm.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL);

        return compileAndInit(dml, dev, DirectMlBindings.DML_OPERATOR_GEMM, gemm, true);
    }

    /**
     * Element-wise SQRT on N scalars with the eps² bias folded in.
     */
    private MiniOp buildBatchedSqrt(MemorySegment dml, MemorySegment dev,
                                    float scale, float bias) throws WindowsNativeException {
        int[] shape = {N, 1, 1, 1};
        long bytes = (long) N * Float.BYTES;
        MemorySegment inDesc = bufferTensorDesc(shape, null, bytes);
        MemorySegment outDesc = bufferTensorDesc(shape, null, bytes);

        MemorySegment scaleBias = arena.allocate(8, 4);
        scaleBias.set(ValueLayout.JAVA_FLOAT, 0, scale);
        scaleBias.set(ValueLayout.JAVA_FLOAT, 4, bias);

        // DML_ELEMENT_WISE_SQRT_OPERATOR_DESC: {InputTensor*, OutputTensor*, DML_SCALE_BIAS*}
        MemorySegment desc = arena.allocate(24, 8);
        desc.set(ValueLayout.ADDRESS, 0, inDesc);
        desc.set(ValueLayout.ADDRESS, 8, outDesc);
        desc.set(ValueLayout.ADDRESS, 16, scaleBias);

        return compileAndInit(dml, dev, DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_SQRT, desc, true);
    }

    /**
     * Broadcast divide: A[N,1,1,H] / B[N,1,1,H] = Y[N,1,1,H]. B has
     * strides {@code [1,0,0,0]} so the H lanes of row k all read the
     * k-th float in the (N-element) backing buffer.
     */
    private MiniOp buildBatchedBroadcastDivide(MemorySegment dml, MemorySegment dev)
            throws WindowsNativeException {
        int[] shape = {N, 1, 1, H};
        long aBytes = (long) N * H * Float.BYTES;
        long nBytes = (long) N * Float.BYTES;
        MemorySegment aDesc = bufferTensorDesc(shape, null, aBytes);
        // Broadcast B: same logical shape, stride[0]=1 (advance one float per row),
        // strides[1..3]=0 (all H lanes read the same per-row scalar).
        MemorySegment bDesc = bufferTensorDesc(shape, new int[]{1, 0, 0, 0}, nBytes);
        MemorySegment oDesc = bufferTensorDesc(shape, null, aBytes);

        // DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC: {ATensor*, BTensor*, OutputTensor*}
        MemorySegment desc = arena.allocate(24, 8);
        desc.set(ValueLayout.ADDRESS, 0, aDesc);
        desc.set(ValueLayout.ADDRESS, 8, bDesc);
        desc.set(ValueLayout.ADDRESS, 16, oDesc);

        return compileAndInit(dml, dev, DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_DIVIDE, desc, true);
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

    // ── dispatch ────────────────────────────────────────────────────────

    /**
     * Normalise every row of {@code x[N,H]} into {@code y[N,H]} in-place
     * w.r.t. the L2 norm of that row (plus the compiled epsilon bias).
     * {@code y} may alias {@code x}.
     */
    public void dispatch(DirectMlTensor x, DirectMlTensor y, float epsilon)
            throws DirectMlRuntimeException {
        ensureOpen();
        validateNH(x, "x");
        validateNH(y, "y");
        if (epsilon != this.epsilon) {
            throw new DirectMlRuntimeException(
                    "DirectMlBatchedL2NormalizeKernel was compiled for epsilon="
                            + this.epsilon + ", caller passed " + epsilon);
        }

        DefaultGpuBuffer xb = unwrap(x.buffer(), "x");
        DefaultGpuBuffer yb = unwrap(y.buffer(), "y");

        try {
            runSumOfSquares(xb);
            runScalarSqrt();
            runBroadcastDivide(xb, yb);
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlBatchedL2NormalizeKernel.dispatch failed" + formatDebugMessages(wb), e);
        }
    }

    private void runSumOfSquares(DefaultGpuBuffer xb) throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();

        try (Arena scratch = Arena.ofConfined()) {
            long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(sumSqOp.descriptorHeap, scratch);
            long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(sumSqOp.descriptorHeap, scratch);
            MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, sumSqOp.compiled,
                    cpuStart, gpuStart, sumSqOp.descriptorCount);
            MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);
            try {
                long xBytes = (long) N * H * Float.BYTES;
                MemorySegment inputs = scratch.allocate(16L * 3, 8);
                setBufferBinding(scratch, inputs, 0, xb.resource(), xBytes);
                setBufferBinding(scratch, inputs, 1, xb.resource(), xBytes);
                inputs.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_BINDING_TYPE_NONE);
                inputs.set(ValueLayout.ADDRESS, 40, MemorySegment.NULL);
                DirectMlBindings.bindInputs(bt, 3, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, sBuf, (long) N * Float.BYTES);
                DirectMlBindings.bindOutputs(bt, 1, outputs);

                bindTempPersist(scratch, bt, sumSqOp);

                MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
                MemorySegment cl = null;
                try {
                    cl = D3D12Bindings.createCommandList(dev,
                            D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);
                    transitionToUav(cl, xb, scratch);
                    transitionInternalToUav(cl, sBuf, () -> sBufState, s -> sBufState = s, scratch);
                    D3D12Bindings.setDescriptorHeaps(cl, sumSqOp.descriptorHeap, scratch);
                    DirectMlBindings.recordDispatch(cmdRecorder, cl, sumSqOp.compiled, bt);
                    D3D12Bindings.executeOrDefer(dev, q, cl, alloc, scratch);
                } finally {
                    if (cl != null) DxgiBindings.release(cl);
                    DxgiBindings.release(alloc);
                }
            } finally {
                DxgiBindings.release(bt);
            }
        }
    }

    private void runScalarSqrt() throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();

        try (Arena scratch = Arena.ofConfined()) {
            long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(sqrtOp.descriptorHeap, scratch);
            long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(sqrtOp.descriptorHeap, scratch);
            MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, sqrtOp.compiled,
                    cpuStart, gpuStart, sqrtOp.descriptorCount);
            MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);
            try {
                MemorySegment inputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, inputs, 0, sBuf, (long) N * Float.BYTES);
                DirectMlBindings.bindInputs(bt, 1, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, nBuf, (long) N * Float.BYTES);
                DirectMlBindings.bindOutputs(bt, 1, outputs);

                bindTempPersist(scratch, bt, sqrtOp);

                MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
                MemorySegment cl = null;
                try {
                    cl = D3D12Bindings.createCommandList(dev,
                            D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);
                    transitionInternalToUav(cl, sBuf, () -> sBufState, s -> sBufState = s, scratch);
                    transitionInternalToUav(cl, nBuf, () -> nBufState, s -> nBufState = s, scratch);
                    D3D12Bindings.setDescriptorHeaps(cl, sqrtOp.descriptorHeap, scratch);
                    DirectMlBindings.recordDispatch(cmdRecorder, cl, sqrtOp.compiled, bt);
                    D3D12Bindings.executeOrDefer(dev, q, cl, alloc, scratch);
                } finally {
                    if (cl != null) DxgiBindings.release(cl);
                    DxgiBindings.release(alloc);
                }
            } finally {
                DxgiBindings.release(bt);
            }
        }
    }

    private void runBroadcastDivide(DefaultGpuBuffer xb, DefaultGpuBuffer yb)
            throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();

        try (Arena scratch = Arena.ofConfined()) {
            long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(divideOp.descriptorHeap, scratch);
            long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(divideOp.descriptorHeap, scratch);
            MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, divideOp.compiled,
                    cpuStart, gpuStart, divideOp.descriptorCount);
            MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);
            try {
                long xBytes = (long) N * H * Float.BYTES;
                MemorySegment inputs = scratch.allocate(16L * 2, 8);
                setBufferBinding(scratch, inputs, 0, xb.resource(), xBytes);
                setBufferBinding(scratch, inputs, 1, nBuf, (long) N * Float.BYTES);
                DirectMlBindings.bindInputs(bt, 2, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, yb.resource(), xBytes);
                DirectMlBindings.bindOutputs(bt, 1, outputs);

                bindTempPersist(scratch, bt, divideOp);

                MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
                MemorySegment cl = null;
                try {
                    cl = D3D12Bindings.createCommandList(dev,
                            D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);
                    transitionToUav(cl, xb, scratch);
                    transitionInternalToUav(cl, nBuf, () -> nBufState, s -> nBufState = s, scratch);
                    transitionToUav(cl, yb, scratch);
                    D3D12Bindings.setDescriptorHeaps(cl, divideOp.descriptorHeap, scratch);
                    DirectMlBindings.recordDispatch(cmdRecorder, cl, divideOp.compiled, bt);
                    D3D12Bindings.executeOrDefer(dev, q, cl, alloc, scratch);
                } finally {
                    if (cl != null) DxgiBindings.release(cl);
                    DxgiBindings.release(alloc);
                }
            } finally {
                DxgiBindings.release(bt);
            }
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
        closeMiniOp(divideOp);
        closeMiniOp(sqrtOp);
        closeMiniOp(sumSqOp);
        if (!nBuf.equals(MemorySegment.NULL)) try {
            DxgiBindings.release(nBuf);
        } catch (Exception ignored) {
        }
        if (!sBuf.equals(MemorySegment.NULL)) try {
            DxgiBindings.release(sBuf);
        } catch (Exception ignored) {
        }
        arena.close();
    }

    private static void closeMiniOp(MiniOp m) {
        if (m == null) return;
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
        if (closed) throw new DirectMlRuntimeException("DirectMlBatchedL2NormalizeKernel already closed");
    }

    private void validateNH(DirectMlTensor t, String name) throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32, was " + t.dataType());
        }
        long expected = (long) N * H;
        if (t.shape().elementCount() != expected) {
            throw new DirectMlRuntimeException(name + " must hold " + expected
                    + " elements (N=" + N + " * H=" + H + "), got " + t.shape().elementCount());
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

    public int batch() {
        return N;
    }

    public int hidden() {
        return H;
    }

    public float epsilon() {
        return epsilon;
    }
}

