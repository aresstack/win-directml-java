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
 * DirectML-L2-Normalisierung über die letzte Dimension eines
 * {@code N}-Float-Vektors:
 * <pre>
 *   y_i = x_i / max(sqrt(Σ x_j²), ε)        approximated by
 *   y_i = x_i / sqrt(Σ x_j² + ε²)
 * </pre>
 * <p>
 * Composite-Implementierung aus drei FL-1.0-Primitiven, die in jedem
 * jemals ausgelieferten {@code DirectML.dll} (Windows 10 RTM, Windows 11
 * In-Box 1.8.0 und neuer) garantiert vorhanden sind:
 * <ol>
 *   <li><b>Sum-of-squares über GEMM</b>
 *       (op {@link DirectMlBindings#DML_OPERATOR_GEMM}): GEMM
 *       {@code A[1,N] · B[N,1] = S[1,1]} mit {@code B = A} und beiden
 *       Transformationen auf {@code DML_MATRIX_TRANSFORM_NONE} – die
 *       Spaltenform entsteht aus dem {@code B}-Tensor-Deskriptor
 *       ({@code [1,1,N,1]}), nicht aus einem Laufzeit-Transpose. In einem
 *       einzigen Dispatch ergibt sich {@code s = Σ x_j²}. Kein expliziter
 *       REDUCE-Operator nötig.</li>
 *   <li><b>{@code sqrt(s + ε²)} via {@link DirectMlBindings#DML_OPERATOR_ELEMENT_WISE_SQRT}</b>
 *       mit gefaltetem {@code DML_SCALE_BIAS(scale=1, bias=ε²)} – ein
 *       weiterer Dispatch auf einem 1-Float-Tensor.</li>
 *   <li><b>Per-Lane-Divide via {@link DirectMlBindings#DML_OPERATOR_ELEMENT_WISE_DIVIDE}</b>
 *       mit Broadcast über Nullstrides:
 *       {@code y[1,1,1,N] = x[1,1,1,N] / n[1,1,1,N(strides=0)]}.
 *       Liest den Normskalar als ein-Element-Broadcast in alle
 *       {@code N} Ausgabe-Lanes.</li>
 * </ol>
 * Damit bleibt der finale Embedding-Vektor bis zum letzten Download
 * komplett auf der GPU; der PCIe-Read-Back schrumpft auf die
 * {@code H = 384} Floats des bereits normierten Outputs.
 * <p>
 * Das Epsilon wird zur Compile-Zeit in den SQRT-Bias gefaltet, daher ist
 * die Instanz an ein festes {@code (N, ε)}-Paar gebunden. {@link #dispatch}
 * verifiziert das übergebene {@code epsilon} und wirft, falls es nicht
 * mit der Compile-Konfiguration übereinstimmt – Anwender, die mehrere
 * Epsilons benötigen, halten mehrere Kernel-Instanzen vor (eine pro
 * Epsilon-Wert).
 * <p>
 * <b>Korrektheitsnote:</b> Die mathematisch exakte Definition
 * {@code y = x / max(|x|, ε)} weicht für extrem kleine Vektoren
 * geringfügig von {@code y = x / sqrt(|x|² + ε²)} ab. Für die hier
 * relevanten Embedding-Vektoren (Norm ≈ 1) ist der Unterschied
 * deutlich unterhalb FP32-Rundungsfehler.
 */
public final class DirectMlL2NormalizeKernel implements L2NormalizeKernel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlL2NormalizeKernel.class);

    private final WindowsBindings wb;
    private final int N;
    private final float epsilon;
    private final Arena arena;

    private final MiniOp sumSqOp;   // GEMM  : s[1,1,1,1] = x · xᵀ
    private final MiniOp sqrtOp;    // SQRT  : n[1,1,1,1] = sqrt(scale·s + bias)
    private final MiniOp divideOp;  // DIVIDE: y[1,1,1,N] = x / broadcast(n)
    private final MemorySegment cmdRecorder;

    // Two scalar UAV buffers carry the sum-of-squares and the norm
    // between dispatches. They are kernel-resident.
    private final MemorySegment sBuf;   // 1 float
    private final MemorySegment nBuf;   // 1 float
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

    public DirectMlL2NormalizeKernel(DirectMlContextImpl ctx, int elementCount, float epsilon)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (elementCount <= 0) {
            throw new IllegalArgumentException("elementCount must be > 0");
        }
        if (!(epsilon >= 0f) || Float.isNaN(epsilon) || Float.isInfinite(epsilon)) {
            throw new IllegalArgumentException("epsilon must be finite and >= 0, was " + epsilon);
        }
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.N = elementCount;
        this.epsilon = epsilon;
        this.arena = Arena.ofShared();

        MiniOp gemmOp = null, sqrtMini = null, divMini = null;
        MemorySegment recorder = MemorySegment.NULL;
        MemorySegment s = MemorySegment.NULL, n = MemorySegment.NULL;
        try {
            MemorySegment dml = wb.getDmlDevice();
            MemorySegment dev = wb.getD3d12Device();
            recorder = DirectMlBindings.createCommandRecorder(dml, arena);

            // ── sub-op 1: GEMM(x[1,N] · xᵀ[N,1]) → s[1,1] ──────────────────
            gemmOp = buildSumOfSquaresGemm(dml, dev);

            // ── sub-op 2: SQRT(scale=1, bias=ε²) on 1-element tensor ──────
            float epsSq = epsilon * epsilon;
            sqrtMini = buildScalarSqrt(dml, dev, 1.0f, epsSq);

            // ── sub-op 3: DIVIDE with broadcast(n) over hidden axis ───────
            divMini = buildBroadcastDivide(dml, dev);

            // ── scalar intermediate buffers ───────────────────────────────
            s = D3D12Bindings.createDefaultBuffer(dev, Float.BYTES, arena);
            n = D3D12Bindings.createDefaultBuffer(dev, Float.BYTES, arena);

            this.sumSqOp = gemmOp;
            this.sqrtOp = sqrtMini;
            this.divideOp = divMini;
            this.cmdRecorder = recorder;
            this.sBuf = s;
            this.nBuf = n;

            log.info("DirectMlL2NormalizeKernel ready: N={}, ε={} (GEMM+SQRT+DIVIDE, FL-1.0)",
                    N, epsilon);
        } catch (WindowsNativeException | RuntimeException e) {
            String dbg = formatDebugMessages(wb);
            if (divMini != null) closeMiniOp(divMini);
            if (sqrtMini != null) closeMiniOp(sqrtMini);
            if (gemmOp != null) closeMiniOp(gemmOp);
            if (!n.equals(MemorySegment.NULL)) try { DxgiBindings.release(n); } catch (Exception ignored) {}
            if (!s.equals(MemorySegment.NULL)) try { DxgiBindings.release(s); } catch (Exception ignored) {}
            if (!recorder.equals(MemorySegment.NULL)) try { DxgiBindings.release(recorder); } catch (Exception ignored) {}
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlL2NormalizeKernel" + dbg, e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Sub-op builders
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Sum-of-squares via GEMM: {@code Y[1,1] = X[1,N] · X[N,1]}. Both
     * GEMM transforms are {@code DML_MATRIX_TRANSFORM_NONE}; the column
     * shape comes from the {@code B}-tensor descriptor itself
     * ({@code [1,1,N,1]}), not from a runtime transpose. A and B alias
     * the same underlying buffer, so a single dispatch yields
     * {@code s = Σ x_j²} without an explicit REDUCE operator.
     */
    private MiniOp buildSumOfSquaresGemm(MemorySegment dml, MemorySegment dev)
            throws WindowsNativeException {
        long xBytes = (long) N * Float.BYTES;
        // Logical A: [1,1,1,N] – the row vector.
        MemorySegment aDesc = bufferTensorDesc(new int[]{1, 1, 1, N}, null, xBytes);
        // Logical B: [1,1,N,1] – the column vector (DML reads the same
        // underlying buffer; with TransB=TRANSPOSE the GEMM treats it as
        // an N×1 matrix). totalSize stays the full N floats.
        MemorySegment bDesc = bufferTensorDesc(new int[]{1, 1, N, 1}, null, xBytes);
        MemorySegment yDesc = bufferTensorDesc(new int[]{1, 1, 1, 1}, null, Float.BYTES);

        // DML_GEMM_OPERATOR_DESC (56 bytes – same layout as DirectMlMeanPoolKernel).
        MemorySegment gemm = arena.allocate(56, 8);
        gemm.set(ValueLayout.ADDRESS, 0, aDesc);
        gemm.set(ValueLayout.ADDRESS, 8, bDesc);
        gemm.set(ValueLayout.ADDRESS, 16, MemorySegment.NULL);                 // CTensor
        gemm.set(ValueLayout.ADDRESS, 24, yDesc);
        gemm.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
        gemm.set(ValueLayout.JAVA_INT, 36, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
        gemm.set(ValueLayout.JAVA_FLOAT, 40, 1.0f);                            // Alpha
        gemm.set(ValueLayout.JAVA_FLOAT, 44, 0.0f);                            // Beta
        gemm.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL);                 // FusedActivation

        return compileAndInit(dml, dev, DirectMlBindings.DML_OPERATOR_GEMM, gemm,
                /* persistOutputs */ true);
    }

    /**
     * Element-wise SQRT on a single-float tensor with the epsilon² bias
     * folded into DML_SCALE_BIAS so a single dispatch yields
     * {@code n = sqrt(s + ε²)}.
     */
    private MiniOp buildScalarSqrt(MemorySegment dml, MemorySegment dev,
                                   float scale, float bias) throws WindowsNativeException {
        int[] shape = {1, 1, 1, 1};
        MemorySegment inDesc = bufferTensorDesc(shape, null, Float.BYTES);
        MemorySegment outDesc = bufferTensorDesc(shape, null, Float.BYTES);

        MemorySegment scaleBias = arena.allocate(8, 4);
        scaleBias.set(ValueLayout.JAVA_FLOAT, 0, scale);
        scaleBias.set(ValueLayout.JAVA_FLOAT, 4, bias);

        // DML_ELEMENT_WISE_SQRT_OPERATOR_DESC: {InputTensor*, OutputTensor*, DML_SCALE_BIAS*}
        MemorySegment desc = arena.allocate(24, 8);
        desc.set(ValueLayout.ADDRESS, 0, inDesc);
        desc.set(ValueLayout.ADDRESS, 8, outDesc);
        desc.set(ValueLayout.ADDRESS, 16, scaleBias);

        return compileAndInit(dml, dev, DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_SQRT, desc,
                /* persistOutputs */ true);
    }

    /**
     * Broadcast-divide: A[1,1,1,N] / B[1,1,1,N(strides=[0,0,0,0])] = Y[1,1,1,N].
     * The B tensor still has logical shape {@code [1,1,1,N]} so the op
     * accepts the binding, but all element strides are zero so DML
     * fetches the same single float for every output lane.
     */
    private MiniOp buildBroadcastDivide(MemorySegment dml, MemorySegment dev)
            throws WindowsNativeException {
        int[] shape = {1, 1, 1, N};
        long aBytes = (long) N * Float.BYTES;
        MemorySegment aDesc = bufferTensorDesc(shape, null, aBytes);
        // Broadcast B: same logical shape, strides=0 in every axis, only
        // one underlying float in the buffer.
        MemorySegment bDesc = bufferTensorDesc(shape, new int[]{0, 0, 0, 0}, Float.BYTES);
        MemorySegment oDesc = bufferTensorDesc(shape, null, aBytes);

        // DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC: {ATensor*, BTensor*, OutputTensor*}
        MemorySegment desc = arena.allocate(24, 8);
        desc.set(ValueLayout.ADDRESS, 0, aDesc);
        desc.set(ValueLayout.ADDRESS, 8, bDesc);
        desc.set(ValueLayout.ADDRESS, 16, oDesc);

        return compileAndInit(dml, dev, DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_DIVIDE, desc,
                /* persistOutputs */ true);
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

    // ─────────────────────────────────────────────────────────────────────
    // Dispatch
    // ─────────────────────────────────────────────────────────────────────

    @Override
    public void dispatch(DirectMlTensor x, DirectMlTensor y, float epsilon)
            throws DirectMlRuntimeException {
        ensureOpen();
        validate(x, "x");
        validate(y, "y");
        // ε is baked into the SQRT bias at compile time; the only way to
        // honour a different ε would be to recompile, which defeats the
        // purpose. Fail loudly rather than silently use the wrong value.
        if (epsilon != this.epsilon) {
            throw new DirectMlRuntimeException(
                    "DirectMlL2NormalizeKernel was compiled for epsilon="
                            + this.epsilon + ", caller passed " + epsilon);
        }

        DefaultGpuBuffer xb = unwrap(x.buffer(), "x");
        DefaultGpuBuffer yb = unwrap(y.buffer(), "y");

        try {
            // 1. s = Σ x²
            runSumOfSquares(xb);
            // 2. n = sqrt(s + ε²)
            runScalarSqrt();
            // 3. y = x / broadcast(n)
            runBroadcastDivide(xb, yb);
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlL2NormalizeKernel.dispatch failed" + formatDebugMessages(wb), e);
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
                // GEMM has 3 input slots (A, B, C). A=B=x, C=none.
                MemorySegment inputs = scratch.allocate(16L * 3, 8);
                setBufferBinding(scratch, inputs, 0, xb.resource(), (long) N * Float.BYTES);
                setBufferBinding(scratch, inputs, 1, xb.resource(), (long) N * Float.BYTES);
                inputs.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_BINDING_TYPE_NONE);
                inputs.set(ValueLayout.ADDRESS, 40, MemorySegment.NULL);
                DirectMlBindings.bindInputs(bt, 3, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, sBuf, Float.BYTES);
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
                    D3D12Bindings.executeAndWait(dev, q, cl, scratch);
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
                setBufferBinding(scratch, inputs, 0, sBuf, Float.BYTES);
                DirectMlBindings.bindInputs(bt, 1, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, nBuf, Float.BYTES);
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
                    D3D12Bindings.executeAndWait(dev, q, cl, scratch);
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
                MemorySegment inputs = scratch.allocate(16L * 2, 8);
                setBufferBinding(scratch, inputs, 0, xb.resource(), (long) N * Float.BYTES);
                setBufferBinding(scratch, inputs, 1, nBuf, Float.BYTES);
                DirectMlBindings.bindInputs(bt, 2, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, yb.resource(), (long) N * Float.BYTES);
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
                    D3D12Bindings.executeAndWait(dev, q, cl, scratch);
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
        try { DxgiBindings.release(cmdRecorder); } catch (Exception ignored) {}
        closeMiniOp(divideOp);
        closeMiniOp(sqrtOp);
        closeMiniOp(sumSqOp);
        if (!nBuf.equals(MemorySegment.NULL)) try { DxgiBindings.release(nBuf); } catch (Exception ignored) {}
        if (!sBuf.equals(MemorySegment.NULL)) try { DxgiBindings.release(sBuf); } catch (Exception ignored) {}
        arena.close();
    }

    private static void closeMiniOp(MiniOp m) {
        if (m == null) return;
        try { DxgiBindings.release(m.descriptorHeap); } catch (Exception ignored) {}
        try { DxgiBindings.release(m.compiled); }      catch (Exception ignored) {}
        if (!m.persistBuffer.equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(m.persistBuffer); } catch (Exception ignored) {}
        }
        if (!m.tempBuffer.equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(m.tempBuffer); }    catch (Exception ignored) {}
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
        if (closed) throw new DirectMlRuntimeException("DirectMlL2NormalizeKernel already closed");
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

    public int elementCount() { return N; }

    public float epsilon() { return epsilon; }
}

