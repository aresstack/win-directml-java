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
 * Scaled-Dot-Product-Attention (SDPA) als <i>Composite</i>-Kernel über
 * vier FL-2.0-DirectML-Primitiven:
 * <ol>
 *   <li><b>Scores</b> = batched-GEMM {@code Q · Kᵀ}, Alpha = {@code scale}
 *       ({@code DML_OPERATOR_GEMM} mit 4-D-Tensoren – die ersten beiden
 *       Dims sind Batch- und Head-Achse)</li>
 *   <li><b>Mask-Add</b> (optional, in-place): {@code scores += mask},
 *       wobei {@code mask} per Broadcast-Strides
 *       {@code [S, 0, 0, 1]} aus {@code [B, 1, 1, S]} auf
 *       {@code [B, H, S, S]} ausgedehnt wird
 *       ({@code DML_OPERATOR_ELEMENT_WISE_ADD})</li>
 *   <li><b>Softmax</b> über die innerste Achse, flach behandelt als
 *       {@code [1, 1, B·H·S, S]}
 *       ({@code DML_OPERATOR_ACTIVATION_SOFTMAX})</li>
 *   <li><b>Output</b> = batched-GEMM {@code probs · V}
 *       ({@code DML_OPERATOR_GEMM})</li>
 * </ol>
 * <p>
 * Alle vier Sub-Operatoren werden im Konstruktor kompiliert und ggf.
 * initialisiert; jeder {@link #dispatch} zeichnet die gesamte Kette in
 * <i>einer</i> Command-List mit UAV-Barrieren zwischen den Stages auf
 * und führt sie mit einem einzigen {@code executeAndWait} aus.
 * <p>
 * Bewusste Entscheidung: <b>kein</b> {@code DML_OPERATOR_MULTIHEAD_ATTENTION}
 * (Enum-ID 164, FL 6.1) als Fast-Path – das setzt eine sehr neue
 * {@code DirectML.dll} voraus. Die Composite-Variante läuft auf jeder
 * heute ausgelieferten DirectML-Runtime, einschließlich der Windows-11-RTM
 * In-Box-Version 1.8.0. Ein FL-6.1-Fast-Path kann später ergänzt werden.
 * <p>
 * Form-Konvention (vom {@link AttentionKernel}-Vertrag):
 * <ul>
 *   <li>{@code Q, K, V}: {@code [B, H, S, D]} (float32, row-major, kontiguous)</li>
 *   <li>{@code mask} (optional): {@code [B, 1, 1, S]}, additiv – Padding
 *       erhält typisch {@code -1e9f}, valide Positionen {@code 0.0f}</li>
 *   <li>{@code Y}: {@code [B, H, S, D]}</li>
 *   <li>{@code scale}: Vorfaktor (üblich {@code 1/sqrt(D)})</li>
 * </ul>
 */
public final class DirectMlAttentionKernel implements AttentionKernel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlAttentionKernel.class);

    private final WindowsBindings wb;
    private final int B;
    private final int H;
    private final int S;
    private final int D;
    private final float scale;
    private final boolean hasMask;
    private final Arena arena;

    private final MemorySegment cmdRecorder;

    // 4-stage pipeline (mask is optional).
    private final Stage stageQk;
    private final Stage stageMask;        // may be null
    private final Stage stageSoftmax;
    private final Stage stageOut;

    // Intermediate score/probability buffer – holds [B,H,S,S] floats.
    private final MemorySegment scoresBuffer;
    private final long scoresBytes;
    private final MemorySegment probsBuffer;
    private final long probsBytes;

    private boolean closed = false;

    /**
     * One DirectML sub-operator (compile + per-op descriptor heap + temp/persist).
     * Lives for the lifetime of the surrounding {@link DirectMlAttentionKernel}.
     */
    private record Stage(MemorySegment compiled, MemorySegment descHeap, int descCount,
                          long tempSize, MemorySegment tempBuffer,
                          long persistSize, MemorySegment persistBuffer) {}

    public DirectMlAttentionKernel(DirectMlContextImpl ctx,
                                   int batch, int heads, int seq, int headDim,
                                   float scale, boolean hasMask)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (batch <= 0 || heads <= 0 || seq <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("batch, heads, seq, headDim must be > 0");
        }
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.B = batch;
        this.H = heads;
        this.S = seq;
        this.D = headDim;
        this.scale = scale;
        this.hasMask = hasMask;
        this.arena = Arena.ofShared();
        this.scoresBytes = (long) B * H * S * S * Float.BYTES;
        this.probsBytes = scoresBytes;

        try {
            MemorySegment dml = wb.getDmlDevice();
            MemorySegment dev = wb.getD3d12Device();
            this.cmdRecorder = DirectMlBindings.createCommandRecorder(dml, arena);

            // ── Stage 1: scores = Q · Kᵀ · scale ──────────────────────────
            //
            // DML_GEMM on 4-D tensors. Leading two dims (B, H) are interpreted
            // as batch dims. TransB = TRANSPOSE means the B-input is treated
            // as [B, H, D, S] for the matmul (last two axes swapped), so
            // matmul yields [B, H, S, S]. Alpha carries the SDPA scale factor.
            this.stageQk = buildGemm(
                    /* aSizes */ new int[]{B, H, S, D},
                    /* aBytes */ (long) B * H * S * D * Float.BYTES,
                    /* bSizes */ new int[]{B, H, S, D},
                    /* bBytes */ (long) B * H * S * D * Float.BYTES,
                    /* transB */ true,
                    /* ySizes */ new int[]{B, H, S, S},
                    /* yBytes */ scoresBytes,
                    /* alpha  */ scale,
                    /* hasBias*/ false);

            // ── Stage 2 (optional): scores += mask ────────────────────────
            //
            // mask is laid out as [B, 1, 1, S] in memory; we describe it
            // virtually as [B, H, S, S] with strides [S, 0, 0, 1] so the
            // same S floats per batch get broadcast across H and over rows.
            // Output writes back into the scores buffer (in-place).
            if (hasMask) {
                this.stageMask = buildElementWiseAdd(
                        /* aSizes  */ new int[]{B, H, S, S},
                        /* aStrides*/ null,
                        /* aBytes  */ scoresBytes,
                        /* bSizes  */ new int[]{B, H, S, S},
                        /* bStrides*/ new int[]{S, 0, 0, 1},
                        /* bBytes  */ (long) B * S * Float.BYTES,
                        /* ySizes  */ new int[]{B, H, S, S},
                        /* yBytes  */ scoresBytes);
            } else {
                this.stageMask = null;
            }

            // ── Stage 3: probs = softmax(scores) over last axis ───────────
            //
            // Flatten [B,H,S,S] to [1,1,B·H·S,S]; softmax normalises over
            // the innermost dim, which is exactly the attention axis.
            int flatRows = B * H * S;
            this.stageSoftmax = buildSoftmax(flatRows, S, scoresBytes);

            // ── Stage 4: y = probs · V ────────────────────────────────────
            this.stageOut = buildGemm(
                    /* aSizes */ new int[]{B, H, S, S},
                    /* aBytes */ probsBytes,
                    /* bSizes */ new int[]{B, H, S, D},
                    /* bBytes */ (long) B * H * S * D * Float.BYTES,
                    /* transB */ false,
                    /* ySizes */ new int[]{B, H, S, D},
                    /* yBytes */ (long) B * H * S * D * Float.BYTES,
                    /* alpha  */ 1.0f,
                    /* hasBias*/ false);

            // ── Intermediate buffers ──────────────────────────────────────
            this.scoresBuffer = D3D12Bindings.createDefaultBuffer(dev, scoresBytes, arena);
            this.probsBuffer = D3D12Bindings.createDefaultBuffer(dev, probsBytes, arena);

            // ── Initialise sub-operators that report a persistent size ────
            initStageIfNeeded(stageQk);
            if (stageMask != null) initStageIfNeeded(stageMask);
            initStageIfNeeded(stageSoftmax);
            initStageIfNeeded(stageOut);

            log.info("DirectMlAttentionKernel ready: B={}, H={}, S={}, D={}, scale={}, hasMask={}",
                    B, H, S, D, scale, hasMask);
        } catch (WindowsNativeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException(
                    "Failed to build DirectMlAttentionKernel" + dbg, e);
        } catch (RuntimeException e) {
            String dbg = formatDebugMessages(wb);
            arena.close();
            throw new DirectMlRuntimeException(
                    "Failed to build DirectMlAttentionKernel" + dbg, e);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Sub-operator construction
    // ══════════════════════════════════════════════════════════════════════

    private Stage buildGemm(int[] aSizes, long aBytes,
                            int[] bSizes, long bBytes,
                            boolean transB,
                            int[] ySizes, long yBytes,
                            float alpha, boolean hasBias) throws WindowsNativeException {
        MemorySegment aDesc = bufferTensorDesc(aSizes, null, aBytes);
        MemorySegment bDesc = bufferTensorDesc(bSizes, null, bBytes);
        MemorySegment cDesc = MemorySegment.NULL; // no bias in attention SDPA core
        MemorySegment yDesc = bufferTensorDesc(ySizes, null, yBytes);

        // DML_GEMM_OPERATOR_DESC layout (56 bytes), see DirectMlLinearKernel.
        MemorySegment gemm = arena.allocate(56, 8);
        gemm.set(ValueLayout.ADDRESS, 0, aDesc);
        gemm.set(ValueLayout.ADDRESS, 8, bDesc);
        gemm.set(ValueLayout.ADDRESS, 16, cDesc);
        gemm.set(ValueLayout.ADDRESS, 24, yDesc);
        gemm.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
        gemm.set(ValueLayout.JAVA_INT, 36, transB
                ? DirectMlBindings.DML_MATRIX_TRANSFORM_TRANSPOSE
                : DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);
        gemm.set(ValueLayout.JAVA_FLOAT, 40, alpha);
        gemm.set(ValueLayout.JAVA_FLOAT, 44, 1.0f); // Beta (unused with C=NULL)
        gemm.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL); // FusedActivation

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_GEMM, gemm);
        return compileAndAllocate(opDesc);
    }

    private Stage buildElementWiseAdd(int[] aSizes, int[] aStrides, long aBytes,
                                      int[] bSizes, int[] bStrides, long bBytes,
                                      int[] ySizes, long yBytes) throws WindowsNativeException {
        MemorySegment aDesc = bufferTensorDesc(aSizes, aStrides, aBytes);
        MemorySegment bDesc = bufferTensorDesc(bSizes, bStrides, bBytes);
        MemorySegment yDesc = bufferTensorDesc(ySizes, null, yBytes);

        // DML_ELEMENT_WISE_ADD_OPERATOR_DESC: A, B, Output (3 × 8 bytes = 24)
        MemorySegment desc = arena.allocate(24, 8);
        desc.set(ValueLayout.ADDRESS, 0, aDesc);
        desc.set(ValueLayout.ADDRESS, 8, bDesc);
        desc.set(ValueLayout.ADDRESS, 16, yDesc);

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_ELEMENT_WISE_ADD, desc);
        return compileAndAllocate(opDesc);
    }

    private Stage buildSoftmax(int numRows, int rowLength, long totalBytes)
            throws WindowsNativeException {
        int[] shape = {1, 1, numRows, rowLength};
        MemorySegment xDesc = bufferTensorDesc(shape, null, totalBytes);
        MemorySegment yDesc = bufferTensorDesc(shape, null, totalBytes);

        // DML_ACTIVATION_SOFTMAX_OPERATOR_DESC: Input + Output (16 bytes)
        MemorySegment desc = arena.allocate(16, 8);
        desc.set(ValueLayout.ADDRESS, 0, xDesc);
        desc.set(ValueLayout.ADDRESS, 8, yDesc);

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_ACTIVATION_SOFTMAX, desc);
        return compileAndAllocate(opDesc);
    }

    /**
     * Compile an operator, query its binding properties, and allocate
     * a per-stage descriptor heap plus optional temp/persist buffers.
     */
    private Stage compileAndAllocate(MemorySegment opDesc) throws WindowsNativeException {
        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();

        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        MemorySegment compiled;
        try {
            compiled = DirectMlBindings.compileOperator(dml, op,
                    DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        } finally {
            DxgiBindings.release(op);
        }

        long[] bp = DirectMlBindings.getBindingProperties(compiled, arena);
        int descCount = Math.max((int) bp[0], 1);
        long tempSize = bp[1];
        long persistSize = bp[2];

        MemorySegment heap = D3D12Bindings.createDescriptorHeap(dev, descCount, arena);
        MemorySegment tempBuf = (tempSize > 0)
                ? D3D12Bindings.createDefaultBuffer(dev, tempSize, arena) : MemorySegment.NULL;
        MemorySegment persistBuf = (persistSize > 0)
                ? D3D12Bindings.createDefaultBuffer(dev, persistSize, arena) : MemorySegment.NULL;
        return new Stage(compiled, heap, descCount, tempSize, tempBuf, persistSize, persistBuf);
    }

    private void initStageIfNeeded(Stage stage) throws WindowsNativeException {
        if (stage.persistSize() == 0) return;
        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();

        MemorySegment initializer = DirectMlBindings.createOperatorInitializer(dml,
                new MemorySegment[]{stage.compiled()}, arena);
        long[] ibp = DirectMlBindings.getBindingProperties(initializer, arena);
        int initDescCount = Math.max((int) ibp[0], 1);
        long initTempSize = ibp[1];

        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(stage.descHeap(), arena);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(stage.descHeap(), arena);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, initializer,
                cpuStart, gpuStart, initDescCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);
        try {
            MemorySegment inputBindings = DirectMlBindings.allocNoneBindingDesc(arena);
            DirectMlBindings.bindInputs(bt, 1, inputBindings);

            MemorySegment outArr = arena.allocate(16, 8);
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena,
                    stage.persistBuffer(), 0, stage.persistSize());
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
                D3D12Bindings.setDescriptorHeaps(cl, stage.descHeap(), arena);
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

    // ══════════════════════════════════════════════════════════════════════
    // Dispatch
    // ══════════════════════════════════════════════════════════════════════

    @Override
    public void dispatch(DirectMlTensor q, DirectMlTensor k, DirectMlTensor v,
                         DirectMlTensor mask, DirectMlTensor y, float dispatchScale)
            throws DirectMlRuntimeException {
        ensureOpen();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment qm = wb.getCommandQueue();
        try (Arena scratch = Arena.ofConfined()) {
            MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
            MemorySegment cl = null;
            try {
                cl = D3D12Bindings.createCommandList(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);
                recordOnto(cl, scratch, q, k, v, mask, y, dispatchScale);
                D3D12Bindings.executeOrDefer(dev, qm, cl, alloc, scratch);
            } finally {
                if (cl != null) DxgiBindings.release(cl);
                DxgiBindings.release(alloc);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlAttentionKernel.dispatch failed" + formatDebugMessages(wb), e);
        }
    }

    /**
     * Records the full 4-stage attention pipeline (Q·Kᵀ → +mask → softmax →
     * ·V) into the caller-supplied command list. Used by encoder coalescing
     * to fold the attention sub-graph into the per-layer command list.
     */
    public void recordOnto(MemorySegment cl, Arena scratch,
                           DirectMlTensor q, DirectMlTensor k, DirectMlTensor v,
                           DirectMlTensor mask, DirectMlTensor y, float dispatchScale)
            throws DirectMlRuntimeException {
        ensureOpen();
        if (dispatchScale != scale) {
            throw new DirectMlRuntimeException("scale mismatch: kernel compiled with "
                    + scale + ", got " + dispatchScale);
        }
        validateQkv(q, "q");
        validateQkv(k, "k");
        validateQkv(v, "v");
        validateQkv(y, "y");
        if (hasMask) {
            if (mask == null) {
                throw new DirectMlRuntimeException(
                        "kernel was built with hasMask=true but mask=null");
            }
            validateMask(mask);
        } else if (mask != null) {
            throw new DirectMlRuntimeException(
                    "kernel was built with hasMask=false but mask!=null");
        }

        DefaultGpuBuffer qb = unwrap(q.buffer(), "q");
        DefaultGpuBuffer kb = unwrap(k.buffer(), "k");
        DefaultGpuBuffer vb = unwrap(v.buffer(), "v");
        DefaultGpuBuffer yb = unwrap(y.buffer(), "y");
        DefaultGpuBuffer mb = (mask != null) ? unwrap(mask.buffer(), "mask") : null;

        long qkvBytes = (long) B * H * S * D * Float.BYTES;

        MemorySegment btQk = null, btMask = null, btSoftmax = null, btOut = null;
        try {
            btQk = buildGemmBindingTable(scratch, stageQk,
                    qb.resource(), qkvBytes,
                    kb.resource(), qkvBytes,
                    scoresBuffer, scoresBytes);

            if (stageMask != null) {
                btMask = buildAddBindingTable(scratch, stageMask,
                        scoresBuffer, scoresBytes,
                        mb.resource(), (long) B * S * Float.BYTES,
                        scoresBuffer, scoresBytes);
            }

            btSoftmax = buildUnaryBindingTable(scratch, stageSoftmax,
                    scoresBuffer, scoresBytes,
                    probsBuffer, probsBytes);

            btOut = buildGemmBindingTable(scratch, stageOut,
                    probsBuffer, probsBytes,
                    vb.resource(), qkvBytes,
                    yb.resource(), qkvBytes);

            // All caller-visible buffers must be UAV before dispatch.
            transitionToUav(cl, qb, scratch);
            transitionToUav(cl, kb, scratch);
            transitionToUav(cl, vb, scratch);
            transitionToUav(cl, yb, scratch);
            if (mb != null) transitionToUav(cl, mb, scratch);

            // Stage 1 – Q · Kᵀ · scale → scores
            D3D12Bindings.setDescriptorHeaps(cl, stageQk.descHeap(), scratch);
            DirectMlBindings.recordDispatch(cmdRecorder, cl, stageQk.compiled(), btQk);
            D3D12Bindings.uavBarrier(cl, scratch);

            // Stage 2 (optional) – scores += mask
            if (stageMask != null) {
                D3D12Bindings.setDescriptorHeaps(cl, stageMask.descHeap(), scratch);
                DirectMlBindings.recordDispatch(cmdRecorder, cl, stageMask.compiled(), btMask);
                D3D12Bindings.uavBarrier(cl, scratch);
            }

            // Stage 3 – probs = softmax(scores)
            D3D12Bindings.setDescriptorHeaps(cl, stageSoftmax.descHeap(), scratch);
            DirectMlBindings.recordDispatch(cmdRecorder, cl, stageSoftmax.compiled(), btSoftmax);
            D3D12Bindings.uavBarrier(cl, scratch);

            // Stage 4 – y = probs · V
            D3D12Bindings.setDescriptorHeaps(cl, stageOut.descHeap(), scratch);
            DirectMlBindings.recordDispatch(cmdRecorder, cl, stageOut.compiled(), btOut);
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "DirectMlAttentionKernel.recordOnto failed" + formatDebugMessages(wb), e);
        } finally {
            if (btQk != null) DxgiBindings.release(btQk);
            if (btMask != null) DxgiBindings.release(btMask);
            if (btSoftmax != null) DxgiBindings.release(btSoftmax);
            if (btOut != null) DxgiBindings.release(btOut);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Per-stage binding-table builders
    // ══════════════════════════════════════════════════════════════════════

    private MemorySegment buildGemmBindingTable(Arena scratch, Stage st,
                                                MemorySegment aRes, long aBytes,
                                                MemorySegment bRes, long bBytes,
                                                MemorySegment yRes, long yBytes)
            throws WindowsNativeException {
        long cpu = D3D12Bindings.getCpuDescriptorHandleForHeapStart(st.descHeap(), scratch);
        long gpu = D3D12Bindings.getGpuDescriptorHandleForHeapStart(st.descHeap(), scratch);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, st.compiled(),
                cpu, gpu, st.descCount());
        MemorySegment bt = DirectMlBindings.createBindingTable(wb.getDmlDevice(), btDesc, scratch);

        // GEMM has 3 input slots (A, B, C). C is bound as NONE.
        MemorySegment inputs = scratch.allocate(16L * 3, 8);
        setBufferBinding(scratch, inputs, 0, aRes, aBytes);
        setBufferBinding(scratch, inputs, 1, bRes, bBytes);
        inputs.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_BINDING_TYPE_NONE);
        inputs.set(ValueLayout.ADDRESS, 40, MemorySegment.NULL);
        DirectMlBindings.bindInputs(bt, 3, inputs);

        MemorySegment outputs = scratch.allocate(16, 8);
        setBufferBinding(scratch, outputs, 0, yRes, yBytes);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(scratch, st, bt);
        return bt;
    }

    private MemorySegment buildAddBindingTable(Arena scratch, Stage st,
                                               MemorySegment aRes, long aBytes,
                                               MemorySegment bRes, long bBytes,
                                               MemorySegment yRes, long yBytes)
            throws WindowsNativeException {
        long cpu = D3D12Bindings.getCpuDescriptorHandleForHeapStart(st.descHeap(), scratch);
        long gpu = D3D12Bindings.getGpuDescriptorHandleForHeapStart(st.descHeap(), scratch);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, st.compiled(),
                cpu, gpu, st.descCount());
        MemorySegment bt = DirectMlBindings.createBindingTable(wb.getDmlDevice(), btDesc, scratch);

        MemorySegment inputs = scratch.allocate(16L * 2, 8);
        setBufferBinding(scratch, inputs, 0, aRes, aBytes);
        setBufferBinding(scratch, inputs, 1, bRes, bBytes);
        DirectMlBindings.bindInputs(bt, 2, inputs);

        MemorySegment outputs = scratch.allocate(16, 8);
        setBufferBinding(scratch, outputs, 0, yRes, yBytes);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(scratch, st, bt);
        return bt;
    }

    private MemorySegment buildUnaryBindingTable(Arena scratch, Stage st,
                                                 MemorySegment xRes, long xBytes,
                                                 MemorySegment yRes, long yBytes)
            throws WindowsNativeException {
        long cpu = D3D12Bindings.getCpuDescriptorHandleForHeapStart(st.descHeap(), scratch);
        long gpu = D3D12Bindings.getGpuDescriptorHandleForHeapStart(st.descHeap(), scratch);
        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, st.compiled(),
                cpu, gpu, st.descCount());
        MemorySegment bt = DirectMlBindings.createBindingTable(wb.getDmlDevice(), btDesc, scratch);

        MemorySegment inputs = scratch.allocate(16, 8);
        setBufferBinding(scratch, inputs, 0, xRes, xBytes);
        DirectMlBindings.bindInputs(bt, 1, inputs);

        MemorySegment outputs = scratch.allocate(16, 8);
        setBufferBinding(scratch, outputs, 0, yRes, yBytes);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(scratch, st, bt);
        return bt;
    }

    private static void bindTempAndPersist(Arena scratch, Stage st, MemorySegment bt) {
        if (st.tempSize() > 0) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(scratch,
                    st.tempBuffer(), 0, st.tempSize());
            DirectMlBindings.bindTemporaryResource(bt,
                    DirectMlBindings.allocBindingDesc(scratch,
                            DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb));
        }
        if (st.persistSize() > 0) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(scratch,
                    st.persistBuffer(), 0, st.persistSize());
            DirectMlBindings.bindPersistentResource(bt,
                    DirectMlBindings.allocBindingDesc(scratch,
                            DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb));
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Lifecycle + helpers
    // ══════════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try { DxgiBindings.release(cmdRecorder); } catch (Exception ignored) {}
        releaseStage(stageQk);
        if (stageMask != null) releaseStage(stageMask);
        releaseStage(stageSoftmax);
        releaseStage(stageOut);
        if (!scoresBuffer.equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(scoresBuffer); } catch (Exception ignored) {}
        }
        if (!probsBuffer.equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(probsBuffer); } catch (Exception ignored) {}
        }
        arena.close();
    }

    private static void releaseStage(Stage st) {
        try { DxgiBindings.release(st.descHeap()); } catch (Exception ignored) {}
        try { DxgiBindings.release(st.compiled()); } catch (Exception ignored) {}
        if (!st.tempBuffer().equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(st.tempBuffer()); } catch (Exception ignored) {}
        }
        if (!st.persistBuffer().equals(MemorySegment.NULL)) {
            try { DxgiBindings.release(st.persistBuffer()); } catch (Exception ignored) {}
        }
    }

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

    private void ensureOpen() throws DirectMlRuntimeException {
        if (closed) throw new DirectMlRuntimeException("DirectMlAttentionKernel already closed");
    }

    private void validateQkv(DirectMlTensor t, String name) throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32, was " + t.dataType());
        }
        long expected = (long) B * H * S * D;
        if (t.shape().elementCount() != expected) {
            throw new DirectMlRuntimeException(name + " expected " + expected
                    + " elements ([B,H,S,D]=[" + B + "," + H + "," + S + "," + D + "]), got "
                    + t.shape().elementCount());
        }
    }

    private void validateMask(DirectMlTensor t) throws DirectMlRuntimeException {
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException("mask must be FLOAT32");
        }
        long expected = (long) B * S;
        if (t.shape().elementCount() != expected) {
            throw new DirectMlRuntimeException("mask expected " + expected
                    + " elements ([B,1,1,S]), got " + t.shape().elementCount());
        }
    }

    private static DefaultGpuBuffer unwrap(GpuBuffer buf, String name)
            throws DirectMlRuntimeException {
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

    public int batch()   { return B; }
    public int heads()   { return H; }
    public int seq()     { return S; }
    public int headDim() { return D; }
    public float scale() { return scale; }
    public boolean hasMask() { return hasMask; }
}

