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
 * BERT-/MiniLM-Style LayerNorm via {@code DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION}.
 * <p>
 * <b>Status:</b> Implementierung in Arbeit. {@code IDMLDevice::CreateOperator}
 * lehnt das aktuelle Operator-Desc-Layout auf dieser DML-Version mit
 * {@code E_INVALIDARG} ab; der zugehörige Test ist temporär {@code @Disabled}.
 * Reaktivierung folgt, sobald der DML-Debug-Layer eingebunden ist und das
 * exakte Feld validiert werden kann. Bis dahin nutzt der Encoder-Pfad die
 * CPU-Fallback-LayerNorm.
 * <p>
 * Normalisiert über die letzte Dimension {@code H} eines Eingangs der Form
 * {@code [M, H]}, intern als {@code [1, 1, M, H]} dargestellt. {@code γ} und
 * {@code β} sind {@code [H]}-Vektoren und werden über die Batch-Dimension
 * {@code M} per Broadcast-Strides {@code [0,0,0,1]} angewendet.
 * <p>
 * Operator-Desc-Layout (DirectML 1.6+):
 * <pre>
 * struct DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC {
 *     const DML_TENSOR_DESC* InputTensor;     // off  0
 *     const DML_TENSOR_DESC* ScaleTensor;     // off  8  (nullable)
 *     const DML_TENSOR_DESC* BiasTensor;      // off 16  (nullable)
 *     const DML_TENSOR_DESC* OutputTensor;    // off 24
 *     const UINT*            Axes;            // off 32
 *     UINT                   AxisCount;       // off 40
 *     BOOL                   NormalizeVariance; // off 44
 *     FLOAT                  Epsilon;         // off 48
 *     // pad 4                                // off 52
 *     const DML_OPERATOR_DESC* FusedActivation; // off 56
 * }; // 64 bytes
 * </pre>
 */
public final class DirectMlLayerNormKernel implements LayerNormKernel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlLayerNormKernel.class);

    private final WindowsBindings wb;
    private final int M;
    private final int H;
    private final float epsilon;
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
     * @param ctx     initialisierter Context
     * @param M       Batch- bzw. Sequenzlänge (Anzahl unabhängiger Zeilen)
     * @param H       Feature-Dimension, über die normalisiert wird
     * @param epsilon numerischer Stabilisator (z. B. 1e-12 für MiniLM/BERT)
     */
    public DirectMlLayerNormKernel(DirectMlContextImpl ctx, int M, int H, float epsilon)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (M <= 0 || H <= 0) {
            throw new IllegalArgumentException("M,H must be > 0");
        }
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.M = M;
        this.H = H;
        this.epsilon = epsilon;
        this.arena = Arena.ofShared();

        try {
            // ── Tensoren (rank 4: [M, 1, H, 1] – MVN0 mit CrossChannel=FALSE
            //   normalisiert per (N,C) über (H,W)=(H,1)=H Elemente → LayerNorm
            //   pro Sequenz-Eintrag M). ──
            MemorySegment xDesc = bufferTensorDesc(new int[]{M, 1, H, 1}, null,
                    (long) M * H * Float.BYTES);
            // γ und β als [1, 1, H, 1] – Broadcast über M.
            MemorySegment gammaDesc = bufferTensorDesc(new int[]{1, 1, H, 1},
                    null, (long) H * Float.BYTES);
            MemorySegment betaDesc = bufferTensorDesc(new int[]{1, 1, H, 1},
                    null, (long) H * Float.BYTES);
            MemorySegment yDesc = bufferTensorDesc(new int[]{M, 1, H, 1}, null,
                    (long) M * H * Float.BYTES);

            // ── DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC (56 bytes) ──
            MemorySegment desc = arena.allocate(56, 8);
            desc.set(ValueLayout.ADDRESS, 0, xDesc);
            desc.set(ValueLayout.ADDRESS, 8, gammaDesc);
            desc.set(ValueLayout.ADDRESS, 16, betaDesc);
            desc.set(ValueLayout.ADDRESS, 24, yDesc);
            desc.set(ValueLayout.JAVA_INT, 32, 0);              // CrossChannel = FALSE
            desc.set(ValueLayout.JAVA_INT, 36, 1);              // NormalizeVariance = TRUE
            desc.set(ValueLayout.JAVA_FLOAT, 40, epsilon);
            desc.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL); // FusedActivation

            MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                    DirectMlBindings.DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION, desc);

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

            log.info("DirectMlLayerNormKernel ready: M={}, H={}, eps={}, desc={}, temp={}B, persist={}B",
                    M, H, epsilon, descriptorCount, tempSize, persistSize);
        } catch (WindowsNativeException e) {
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlLayerNormKernel", e);
        } catch (RuntimeException e) {
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlLayerNormKernel", e);
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
                MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, persistBuffer, 0, persistSize);
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
                D3D12Bindings.setDescriptorHeaps(cl, descriptorHeap, arena);
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
    public void dispatch(DirectMlTensor x, DirectMlTensor gamma, DirectMlTensor beta,
                         DirectMlTensor y, float eps) throws DirectMlRuntimeException {
        ensureOpen();
        if (eps != epsilon) {
            // Epsilon ist im Operator einkompiliert. Falls der Caller einen
            // anderen Wert braucht, muss ein neuer Kernel erzeugt werden.
            throw new DirectMlRuntimeException(
                    "epsilon mismatch: kernel compiled with " + epsilon + ", got " + eps);
        }
        validate2D(x, M, H, "x");
        validate2D(y, M, H, "y");
        validate1D(gamma, H, "gamma");
        validate1D(beta, H, "beta");

        DefaultGpuBuffer xb = unwrap(x.buffer(), "x");
        DefaultGpuBuffer gb = unwrap(gamma.buffer(), "gamma");
        DefaultGpuBuffer bb = unwrap(beta.buffer(), "beta");
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
                MemorySegment inputs = scratch.allocate(16L * 3, 8);
                setBufferBinding(scratch, inputs, 0, xb.resource(), (long) M * H * Float.BYTES);
                setBufferBinding(scratch, inputs, 1, gb.resource(), (long) H * Float.BYTES);
                setBufferBinding(scratch, inputs, 2, bb.resource(), (long) H * Float.BYTES);
                DirectMlBindings.bindInputs(bt, 3, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, yb.resource(), (long) M * H * Float.BYTES);
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
                    transitionToUav(cl, gb, scratch);
                    transitionToUav(cl, bb, scratch);
                    transitionToUav(cl, yb, scratch);
                    D3D12Bindings.setDescriptorHeaps(cl, descriptorHeap, scratch);
                    DirectMlBindings.recordDispatch(cmdRecorder, cl, compiled, bt);
                    D3D12Bindings.executeAndWait(dev, q, cl, scratch);
                } finally {
                    if (cl != null) DxgiBindings.release(cl);
                    DxgiBindings.release(alloc);
                }
            } finally {
                DxgiBindings.release(bt);
            }
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException("DirectMlLayerNormKernel.dispatch failed", e);
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
        if (closed) throw new DirectMlRuntimeException("DirectMlLayerNormKernel already closed");
    }

    private static void validate2D(DirectMlTensor t, int d0, int d1, String name)
            throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32");
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
            throw new DirectMlRuntimeException(name + " must be FLOAT32");
        }
        if (t.shape().elementCount() != n) {
            throw new DirectMlRuntimeException(name + " must hold " + n
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

    public int M() {
        return M;
    }

    public int H() {
        return H;
    }

    public float epsilon() {
        return epsilon;
    }
}

