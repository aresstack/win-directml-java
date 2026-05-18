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

/**
 * Erste produktive DirectML-Kernel-Implementierung:
 * dichte Matrixmultiplikation {@code y = x · W^T + b} über
 * {@code DML_OPERATOR_GEMM}.
 * <p>
 * Form-Konvention (vom {@link LinearKernel}-Vertrag):
 * <ul>
 *   <li>{@code x}: {@code [M, K]} – als DML-A-Tensor {@code [1,1,M,K]}</li>
 *   <li>{@code W}: {@code [N, K]} – als DML-B-Tensor {@code [1,1,N,K]} mit
 *       {@code TransB = TRANSPOSE}</li>
 *   <li>{@code b}: {@code [N]} – als DML-C-Tensor {@code [1,1,M,N]} mit
 *       Broadcast-Strides {@code [0,0,0,1]} (optional)</li>
 *   <li>{@code y}: {@code [M, N]} – als DML-Output {@code [1,1,M,N]}</li>
 * </ul>
 * <p>
 * Lebenszyklus: Compile/Initialize einmal im Konstruktor; jede
 * {@link #dispatch(DirectMlTensor, DirectMlTensor, DirectMlTensor, DirectMlTensor)}
 * baut eine frische Binding-Table und feuert eine eigene Command-List.
 * {@link #close()} gibt alle persistenten DirectML-Ressourcen frei.
 */
public final class DirectMlLinearKernel implements LinearKernel, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(DirectMlLinearKernel.class);

    private final DirectMlContextImpl ctx;
    private final WindowsBindings wb;
    private final int M;
    private final int K;
    private final int N;
    private final boolean hasBias;

    // Per-Kernel-Arena für FFM-Allokationen, die so lange leben wie der Kernel.
    private final Arena arena;

    private final MemorySegment compiled;
    private final MemorySegment cmdRecorder;
    private final MemorySegment descriptorHeap;
    private final int descriptorCount;
    private final long tempSize;
    private final long persistSize;
    private final MemorySegment tempBuffer;   // may be NULL
    private final MemorySegment persistBuffer; // may be NULL

    private boolean closed = false;

    public DirectMlLinearKernel(DirectMlContextImpl ctx, int M, int K, int N, boolean hasBias)
            throws DirectMlRuntimeException {
        if (ctx == null || !ctx.isReady()) {
            throw new DirectMlRuntimeException("Context not ready");
        }
        if (M <= 0 || K <= 0 || N <= 0) {
            throw new IllegalArgumentException("M,K,N must be > 0");
        }
        this.ctx = ctx;
        this.wb = ctx.bindings();
        if (!wb.hasDirectMl()) {
            throw new DirectMlRuntimeException("Context has no DirectML device");
        }
        this.M = M;
        this.K = K;
        this.N = N;
        this.hasBias = hasBias;
        this.arena = Arena.ofShared();

        try {
            // 1) Build DML_GEMM_OPERATOR_DESC and compile.
            MemorySegment aDesc = bufferTensorDesc(new int[]{1, 1, M, K}, null,
                    (long) M * K * Float.BYTES);
            MemorySegment bDesc = bufferTensorDesc(new int[]{1, 1, N, K}, null,
                    (long) N * K * Float.BYTES);
            MemorySegment cDesc = MemorySegment.NULL;
            if (hasBias) {
                // [1,1,M,N] tensor, strides [0,0,0,1] → broadcast a vector of N
                // floats across M rows. Backing buffer holds N floats.
                int[] cSizes = new int[]{1, 1, M, N};
                int[] cStrides = new int[]{0, 0, 0, 1};
                cDesc = bufferTensorDesc(cSizes, cStrides, (long) N * Float.BYTES);
            }
            MemorySegment yDesc = bufferTensorDesc(new int[]{1, 1, M, N}, null,
                    (long) M * N * Float.BYTES);

            MemorySegment gemm = arena.allocate(56, 8);
            gemm.set(java.lang.foreign.ValueLayout.ADDRESS, 0, aDesc);
            gemm.set(java.lang.foreign.ValueLayout.ADDRESS, 8, bDesc);
            gemm.set(java.lang.foreign.ValueLayout.ADDRESS, 16, cDesc);
            gemm.set(java.lang.foreign.ValueLayout.ADDRESS, 24, yDesc);
            gemm.set(java.lang.foreign.ValueLayout.JAVA_INT, 32,
                    DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);   // TransA
            gemm.set(java.lang.foreign.ValueLayout.JAVA_INT, 36,
                    DirectMlBindings.DML_MATRIX_TRANSFORM_TRANSPOSE); // TransB
            gemm.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 40, 1.0f); // Alpha
            gemm.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, 44, 1.0f); // Beta
            gemm.set(java.lang.foreign.ValueLayout.ADDRESS, 48, MemorySegment.NULL); // FusedActivation

            MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                    DirectMlBindings.DML_OPERATOR_GEMM, gemm);
            MemorySegment dml = wb.getDmlDevice();
            MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
            this.compiled = DirectMlBindings.compileOperator(dml, op,
                    DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
            DxgiBindings.release(op);

            // 2) Binding properties (descriptors, temp/persist sizes).
            long[] bp = DirectMlBindings.getBindingProperties(compiled, arena);
            this.descriptorCount = Math.max((int) bp[0], 1);
            this.tempSize = bp[1];
            this.persistSize = bp[2];

            MemorySegment dev = wb.getD3d12Device();
            this.descriptorHeap = D3D12Bindings.createDescriptorHeap(dev, descriptorCount, arena);
            this.cmdRecorder = DirectMlBindings.createCommandRecorder(dml, arena);

            this.tempBuffer = (tempSize > 0)
                    ? D3D12Bindings.createDefaultBuffer(dev, tempSize, arena)
                    : MemorySegment.NULL;
            this.persistBuffer = (persistSize > 0)
                    ? D3D12Bindings.createDefaultBuffer(dev, persistSize, arena)
                    : MemorySegment.NULL;

            // 3) Initializer pass (only if there is persistent state).
            if (persistSize > 0) {
                initializeOperator();
            }

            log.info("DirectMlLinearKernel ready: M={}, K={}, N={}, hasBias={}, "
                            + "desc={}, temp={}B, persist={}B",
                    M, K, N, hasBias, descriptorCount, tempSize, persistSize);
        } catch (WindowsNativeException e) {
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlLinearKernel", e);
        } catch (RuntimeException e) {
            arena.close();
            throw new DirectMlRuntimeException("Failed to build DirectMlLinearKernel", e);
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
            // One inner-op → one DML_BINDING_DESC of type NONE.
            MemorySegment inputBindings = DirectMlBindings.allocNoneBindingDesc(arena);
            DirectMlBindings.bindInputs(bt, 1, inputBindings);

            // The initializer writes the persistent buffer to the operator's
            // outputs slot.
            if (persistSize > 0) {
                MemorySegment outArr = arena.allocate(16, 8);
                MemorySegment bb = DirectMlBindings.allocBufferBinding(arena,
                        persistBuffer, 0, persistSize);
                outArr.set(java.lang.foreign.ValueLayout.JAVA_INT, 0,
                        DirectMlBindings.DML_BINDING_TYPE_BUFFER);
                outArr.set(java.lang.foreign.ValueLayout.ADDRESS, 8, bb);
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
    public void dispatch(DirectMlTensor x, DirectMlTensor weight, DirectMlTensor bias,
                         DirectMlTensor y) throws DirectMlRuntimeException {
        ensureOpen();
        validateShape(x, M, K, "x");
        validateShape(weight, N, K, "weight");
        if (hasBias) {
            if (bias == null) {
                throw new DirectMlRuntimeException("Kernel was built with hasBias=true but bias=null");
            }
            validateBias(bias);
        } else if (bias != null) {
            throw new DirectMlRuntimeException("Kernel was built with hasBias=false but bias!=null");
        }
        validateShape(y, M, N, "y");

        // Ensure all bound resources are in UAV state. Inputs that came from
        // upload() are already UAV; freshly allocated outputs may still be in
        // COMMON. We patch this up with explicit transitions before dispatch.
        DefaultGpuBuffer xb = unwrap(x.buffer(), "x");
        DefaultGpuBuffer wb_ = unwrap(weight.buffer(), "weight");
        DefaultGpuBuffer yb = unwrap(y.buffer(), "y");
        DefaultGpuBuffer bb = (bias != null) ? unwrap(bias.buffer(), "bias") : null;

        MemorySegment dml = wb.getDmlDevice();
        MemorySegment dev = wb.getD3d12Device();
        MemorySegment q = wb.getCommandQueue();

        try (Arena scratch = Arena.ofConfined()) {
            // Binding table on the kernel's descriptor heap.
            long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, scratch);
            long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, scratch);
            MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(scratch, compiled,
                    cpuStart, gpuStart, descriptorCount);
            MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, scratch);

            try {
                // DML_OPERATOR_GEMM has 3 fixed input slots (A, B, C). The C
                // slot must always be present – if the operator was built with
                // CTensor=NULL we bind it as DML_BINDING_TYPE_NONE.
                MemorySegment inputs = scratch.allocate(16L * 3, 8);
                setBufferBinding(scratch, inputs, 0, xb.resource(), (long) M * K * Float.BYTES);
                setBufferBinding(scratch, inputs, 1, wb_.resource(), (long) N * K * Float.BYTES);
                if (hasBias) {
                    setBufferBinding(scratch, inputs, 2, bb.resource(), (long) N * Float.BYTES);
                } else {
                    inputs.set(java.lang.foreign.ValueLayout.JAVA_INT, 32,
                            DirectMlBindings.DML_BINDING_TYPE_NONE);
                    inputs.set(java.lang.foreign.ValueLayout.ADDRESS, 40, MemorySegment.NULL);
                }
                DirectMlBindings.bindInputs(bt, 3, inputs);

                MemorySegment outputs = scratch.allocate(16, 8);
                setBufferBinding(scratch, outputs, 0, yb.resource(), (long) M * N * Float.BYTES);
                DirectMlBindings.bindOutputs(bt, 1, outputs);

                if (tempSize > 0) {
                    MemorySegment bbTmp = DirectMlBindings.allocBufferBinding(scratch,
                            tempBuffer, 0, tempSize);
                    DirectMlBindings.bindTemporaryResource(bt,
                            DirectMlBindings.allocBindingDesc(scratch,
                                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bbTmp));
                }
                if (persistSize > 0) {
                    MemorySegment bbPer = DirectMlBindings.allocBufferBinding(scratch,
                            persistBuffer, 0, persistSize);
                    DirectMlBindings.bindPersistentResource(bt,
                            DirectMlBindings.allocBindingDesc(scratch,
                                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bbPer));
                }

                MemorySegment alloc = D3D12Bindings.createCommandAllocator(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, scratch);
                MemorySegment cl = null;
                try {
                    cl = D3D12Bindings.createCommandList(dev,
                            D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, scratch);

                    // Promote freshly allocated output to UAV before dispatch.
                    transitionToUav(cl, yb, scratch);
                    transitionToUav(cl, xb, scratch);
                    transitionToUav(cl, wb_, scratch);
                    if (bb != null) transitionToUav(cl, bb, scratch);

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
            throw new DirectMlRuntimeException("DirectMlLinearKernel.dispatch failed", e);
        }
    }

    private static void transitionToUav(MemorySegment cl, DefaultGpuBuffer buf, Arena scratch) {
        int state = buf.currentResourceState();
        if (state != D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12Bindings.transitionBarrier(cl, buf.resource(), state,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS, scratch);
            buf.setCurrentResourceState(D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            DxgiBindings.release(cmdRecorder);
        } catch (Exception ignored) { /* close is best-effort */ }
        try {
            DxgiBindings.release(descriptorHeap);
        } catch (Exception ignored) { /* close is best-effort */ }
        try {
            DxgiBindings.release(compiled);
        } catch (Exception ignored) { /* close is best-effort */ }
        if (!persistBuffer.equals(MemorySegment.NULL)) {
            try {
                DxgiBindings.release(persistBuffer);
            } catch (Exception ignored) { /* best-effort */ }
        }
        if (!tempBuffer.equals(MemorySegment.NULL)) {
            try {
                DxgiBindings.release(tempBuffer);
            } catch (Exception ignored) { /* best-effort */ }
        }
        arena.close();
    }

    private void ensureOpen() throws DirectMlRuntimeException {
        if (closed) throw new DirectMlRuntimeException("DirectMlLinearKernel already closed");
    }

    private static void validateShape(DirectMlTensor t, int d0, int d1, String name)
            throws DirectMlRuntimeException {
        if (t == null) throw new DirectMlRuntimeException(name + " tensor is null");
        if (t.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException(name + " must be FLOAT32, was " + t.dataType());
        }
        int rank = t.shape().rank();
        int dim0 = (rank >= 2) ? t.shape().dim(rank - 2) : 1;
        int dim1 = t.shape().dim(rank - 1);
        if (dim0 != d0 || dim1 != d1) {
            throw new DirectMlRuntimeException(name + " expected shape ["
                    + d0 + "," + d1 + "], got " + t.shape());
        }
    }

    private void validateBias(DirectMlTensor b) throws DirectMlRuntimeException {
        if (b.dataType() != TensorDataType.FLOAT32) {
            throw new DirectMlRuntimeException("bias must be FLOAT32");
        }
        if (b.shape().elementCount() != N) {
            throw new DirectMlRuntimeException(
                    "bias must hold " + N + " elements, got " + b.shape().elementCount());
        }
    }

    private static DefaultGpuBuffer unwrap(GpuBuffer buf, String name) throws DirectMlRuntimeException {
        if (!(buf instanceof DefaultGpuBuffer d)) {
            throw new DirectMlRuntimeException(name
                    + " buffer must be DefaultGpuBuffer (got "
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
        array.set(java.lang.foreign.ValueLayout.JAVA_INT, off,
                DirectMlBindings.DML_BINDING_TYPE_BUFFER);
        array.set(java.lang.foreign.ValueLayout.ADDRESS, off + 8, bb);
    }
}

