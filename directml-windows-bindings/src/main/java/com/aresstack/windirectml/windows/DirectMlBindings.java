package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Java 21 FFM bindings for {@code DirectML.dll} – device creation and operator pipeline.
 * <p>
 * Calls directly into the Windows DirectML system DLL.
 * No third-party runtime (no ONNX Runtime, no wrapper libraries).
 */
public final class DirectMlBindings {

    private static final Logger log = LoggerFactory.getLogger(DirectMlBindings.class);

    private DirectMlBindings() {}

    // ── DML constants ────────────────────────────────────────────────────
    public static final int DML_CREATE_DEVICE_FLAG_NONE  = 0;
    public static final int DML_CREATE_DEVICE_FLAG_DEBUG = 1;
    public static final int DML_EXECUTION_FLAG_NONE = 0;

    // DML_TENSOR_DATA_TYPE
    public static final int DML_TENSOR_DATA_TYPE_FLOAT32 = 1;

    // DML_TENSOR_TYPE
    public static final int DML_TENSOR_TYPE_INVALID = 0;
    public static final int DML_TENSOR_TYPE_BUFFER = 1;

    // DML_TENSOR_FLAGS
    public static final int DML_TENSOR_FLAG_NONE = 0;

    // DML_OPERATOR_TYPE values (from DirectML.h – Windows SDK 10.0.26100.0)
    public static final int DML_OPERATOR_ELEMENT_WISE_IDENTITY = 1;
    public static final int DML_OPERATOR_ELEMENT_WISE_ADD        = 4;
    public static final int DML_OPERATOR_BATCH_NORMALIZATION   = 29;
    public static final int DML_OPERATOR_ACTIVATION_RELU       = 44;
    public static final int DML_OPERATOR_CONVOLUTION            = 53;
    public static final int DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION  = 39;
    public static final int DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1 = 50;
    public static final int DML_OPERATOR_GEMM                   = 54;
    public static final int DML_OPERATOR_MAX_POOLING             = 58;

    // DML_CONVOLUTION_MODE / DIRECTION
    public static final int DML_CONVOLUTION_MODE_CROSS_CORRELATION = 0;
    public static final int DML_CONVOLUTION_DIRECTION_FORWARD      = 0;

    // DML_MATRIX_TRANSFORM
    public static final int DML_MATRIX_TRANSFORM_NONE      = 0;
    public static final int DML_MATRIX_TRANSFORM_TRANSPOSE = 1;

    // DML_BINDING_TYPE
    public static final int DML_BINDING_TYPE_NONE   = 0;
    public static final int DML_BINDING_TYPE_BUFFER = 1;

    // ── IDMLDevice vtable slots ──────────────────────────────────────────
    // IUnknown: 0-2, IDMLObject: 3-6
    static final int DML_DEV_CHECK_FEATURE_SUPPORT       = 7;
    static final int DML_DEV_CREATE_OPERATOR              = 8;
    static final int DML_DEV_COMPILE_OPERATOR             = 9;
    static final int DML_DEV_CREATE_OPERATOR_INITIALIZER  = 10;
    static final int DML_DEV_CREATE_COMMAND_RECORDER      = 11;
    static final int DML_DEV_CREATE_BINDING_TABLE         = 12;

    // IDMLDispatchable: GetBindingProperties = slot 8
    // Inheritance: IUnknown(0-2) → IDMLObject(3-6) → IDMLDeviceChild::GetDevice(7) → IDMLPageable(none) → IDMLDispatchable(8)
    static final int DISPATCHABLE_GET_BINDING_PROPERTIES = 8;

    // IDMLCommandRecorder: RecordDispatch = slot 8
    // Inheritance: IUnknown(0-2) → IDMLObject(3-6) → IDMLDeviceChild::GetDevice(7) → IDMLPageable(none) → IDMLCommandRecorder(8)
    static final int RECORDER_RECORD_DISPATCH = 8;

    // IDMLBindingTable inherits from IDMLDeviceChild (NOT just IDMLObject!)
    // Inheritance: IUnknown(0-2) → IDMLObject(3-6) → IDMLDeviceChild::GetDevice(7) → IDMLBindingTable(8+)
    static final int BT_BIND_INPUTS     = 8;
    static final int BT_BIND_OUTPUTS    = 9;
    static final int BT_BIND_TEMPORARY  = 10;
    static final int BT_BIND_PERSISTENT = 11;

    // ── DMLCreateDevice function handle ──────────────────────────────────
    private static final FunctionDescriptor DML_CREATE_DEVICE_DESC =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

    private static volatile MethodHandle dmlCreateDeviceHandle;

    private static MethodHandle getDmlCreateDeviceHandle() {
        if (dmlCreateDeviceHandle == null) {
            synchronized (DirectMlBindings.class) {
                if (dmlCreateDeviceHandle == null) {
                    SymbolLookup dml = SymbolLookup.libraryLookup("DirectML.dll", Arena.global());
                    MemorySegment addr = dml.find("DMLCreateDevice")
                            .orElseThrow(() -> new UnsatisfiedLinkError("DMLCreateDevice not found"));
                    dmlCreateDeviceHandle = Linker.nativeLinker()
                            .downcallHandle(addr, DML_CREATE_DEVICE_DESC);
                }
            }
        }
        return dmlCreateDeviceHandle;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Device creation
    // ══════════════════════════════════════════════════════════════════════

    public static MemorySegment createDevice(MemorySegment d3d12Device, int flags, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_IDMLDevice_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            int hr = (int) getDmlCreateDeviceHandle().invokeExact(d3d12Device, flags, riid, pp);
            HResult.check(hr, "DMLCreateDevice");
            MemorySegment dev = pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
            log.info("IDMLDevice created: {}", dev);
            return dev;
        } catch (WindowsNativeException e) { throw e; }
        catch (Throwable t) { throw new WindowsNativeException("DMLCreateDevice failed", t); }
    }

    public static boolean isAvailable() {
        try { getDmlCreateDeviceHandle(); return true; }
        catch (Exception | UnsatisfiedLinkError e) { return false; }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Operator creation & compilation
    // ══════════════════════════════════════════════════════════════════════

    /**
     * IDMLDevice::CreateOperator (vtable slot 8).
     * @param dmlDevice   IDMLDevice COM pointer
     * @param opDesc      pointer to filled DML_OPERATOR_DESC struct
     * @param arena       arena for allocations
     * @return IDMLOperator COM pointer
     */
    public static MemorySegment createOperator(MemorySegment dmlDevice, MemorySegment opDesc,
                                                Arena arena) throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_IDMLOperator_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(dmlDevice, DML_DEV_CREATE_OPERATOR,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(dmlDevice, opDesc, riid, pp);
            HResult.check(hr, "IDMLDevice::CreateOperator");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) { throw e; }
        catch (Throwable t) { throw new WindowsNativeException("CreateOperator failed", t); }
    }

    /**
     * IDMLDevice::CompileOperator (vtable slot 9).
     * @param dmlDevice   IDMLDevice
     * @param dmlOperator IDMLOperator to compile
     * @param flags       DML_EXECUTION_FLAGS
     * @param arena       arena
     * @return IDMLCompiledOperator COM pointer (also an IDMLDispatchable)
     */
    public static MemorySegment compileOperator(MemorySegment dmlDevice, MemorySegment dmlOperator,
                                                 int flags, Arena arena) throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_IDMLCompiledOperator_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(dmlDevice, DML_DEV_COMPILE_OPERATOR,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(dmlDevice, dmlOperator, flags, riid, pp);
            HResult.check(hr, "IDMLDevice::CompileOperator");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) { throw e; }
        catch (Throwable t) { throw new WindowsNativeException("CompileOperator failed", t); }
    }

    /**
     * IDMLDevice::CreateOperatorInitializer (vtable slot 10).
     */
    public static MemorySegment createOperatorInitializer(MemorySegment dmlDevice,
                                                           MemorySegment[] compiledOps,
                                                           Arena arena) throws WindowsNativeException {
        try {
            MemorySegment opsArray = arena.allocate(ValueLayout.ADDRESS.byteSize() * compiledOps.length, 8);
            for (int i = 0; i < compiledOps.length; i++) {
                opsArray.setAtIndex(ValueLayout.ADDRESS, i, compiledOps[i]);
            }
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_IDMLOperatorInitializer_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(dmlDevice, DML_DEV_CREATE_OPERATOR_INITIALIZER,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(dmlDevice, compiledOps.length, opsArray, riid, pp);
            HResult.check(hr, "IDMLDevice::CreateOperatorInitializer");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) { throw e; }
        catch (Throwable t) { throw new WindowsNativeException("CreateOperatorInitializer failed", t); }
    }

    /**
     * IDMLDevice::CreateCommandRecorder (vtable slot 11).
     */
    public static MemorySegment createCommandRecorder(MemorySegment dmlDevice, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_IDMLCommandRecorder_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(dmlDevice, DML_DEV_CREATE_COMMAND_RECORDER,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(dmlDevice, riid, pp);
            HResult.check(hr, "IDMLDevice::CreateCommandRecorder");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) { throw e; }
        catch (Throwable t) { throw new WindowsNativeException("CreateCommandRecorder failed", t); }
    }

    /**
     * IDMLDevice::CreateBindingTable (vtable slot 12).
     * @param bindingTableDesc pointer to DML_BINDING_TABLE_DESC (can be NULL for deferred init)
     */
    public static MemorySegment createBindingTable(MemorySegment dmlDevice,
                                                    MemorySegment bindingTableDesc,
                                                    Arena arena) throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_IDMLBindingTable_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(dmlDevice, DML_DEV_CREATE_BINDING_TABLE,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(dmlDevice, bindingTableDesc, riid, pp);
            HResult.check(hr, "IDMLDevice::CreateBindingTable");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) { throw e; }
        catch (Throwable t) { throw new WindowsNativeException("CreateBindingTable failed", t); }
    }

    // ══════════════════════════════════════════════════════════════════════
    // IDMLDispatchable operations
    // ══════════════════════════════════════════════════════════════════════

    /**
     * IDMLDispatchable::GetBindingProperties (vtable slot 8).
     * <p>
     * Returns DML_BINDING_PROPERTIES struct (24 bytes on x64).
     * <p>
     * DirectML uses an explicit output-pointer convention (like D3D12's
     * GetCPUDescriptorHandleForHeapStart), NOT the standard x64 sret convention.
     * The actual ABI is: {@code void GetBindingProperties(this, DML_BINDING_PROPERTIES* pResult)}
     * where RCX = this, RDX = output pointer.
     *
     * @return long[3]: {requiredDescriptorCount, temporaryResourceSize, persistentResourceSize}
     */
    public static long[] getBindingProperties(MemorySegment dispatchable, Arena arena) {
        try {
            // DML_BINDING_PROPERTIES: UINT(4) + pad(4) + UINT64(8) + UINT64(8) = 24 bytes
            MemorySegment outBuf = arena.allocate(24, 8);

            // DirectML ABI: fn(this, pResult) — RCX=this, RDX=output pointer
            MethodHandle mh = DxgiBindings.vtableMethod(dispatchable,
                    DISPATCHABLE_GET_BINDING_PROPERTIES,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            mh.invokeExact(dispatchable, outBuf);

            int descCount = outBuf.get(ValueLayout.JAVA_INT, 0);
            long tempSize = outBuf.get(ValueLayout.JAVA_LONG, 8);
            long persistSize = outBuf.get(ValueLayout.JAVA_LONG, 16);
            log.debug("BindingProperties: descCount={}, temp={}, persist={}", descCount, tempSize, persistSize);
            return new long[]{ descCount, tempSize, persistSize };
        } catch (Throwable t) {
            throw new RuntimeException("GetBindingProperties failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // IDMLCommandRecorder
    // ══════════════════════════════════════════════════════════════════════

    /**
     * IDMLCommandRecorder::RecordDispatch (vtable slot 8).
     */
    public static void recordDispatch(MemorySegment recorder, MemorySegment cmdList,
                                       MemorySegment dispatchable, MemorySegment bindingTable) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(recorder, RECORDER_RECORD_DISPATCH,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            mh.invokeExact(recorder, cmdList, dispatchable, bindingTable);
        } catch (Throwable t) {
            throw new RuntimeException("RecordDispatch failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // IDMLBindingTable
    // ══════════════════════════════════════════════════════════════════════

    /** IDMLBindingTable::BindInputs (slot 8). */
    public static void bindInputs(MemorySegment bt, int count, MemorySegment bindings) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, BT_BIND_INPUTS,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            mh.invokeExact(bt, count, bindings);
        } catch (Throwable t) { throw new RuntimeException("BindInputs failed", t); }
    }

    /** IDMLBindingTable::BindOutputs (slot 9). */
    public static void bindOutputs(MemorySegment bt, int count, MemorySegment bindings) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, BT_BIND_OUTPUTS,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            mh.invokeExact(bt, count, bindings);
        } catch (Throwable t) { throw new RuntimeException("BindOutputs failed", t); }
    }

    /** IDMLBindingTable::BindTemporaryResource (slot 10). */
    public static void bindTemporaryResource(MemorySegment bt, MemorySegment binding) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, BT_BIND_TEMPORARY,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            mh.invokeExact(bt, binding);
        } catch (Throwable t) { throw new RuntimeException("BindTemporary failed", t); }
    }

    /** IDMLBindingTable::BindPersistentResource (slot 11). */
    public static void bindPersistentResource(MemorySegment bt, MemorySegment binding) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, BT_BIND_PERSISTENT,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            mh.invokeExact(bt, binding);
        } catch (Throwable t) { throw new RuntimeException("BindPersistent failed", t); }
    }

    /** IDMLBindingTable::Reset (slot 12). */
    public static void resetBindingTable(MemorySegment bt, MemorySegment newDesc) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, 12,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(bt, newDesc);
            HResult.check(hr, "IDMLBindingTable::Reset");
        } catch (Throwable t) { throw new RuntimeException("BindingTable::Reset failed", t); }
    }

    // ══════════════════════════════════════════════════════════════════════
    // DML struct builders (native memory layout helpers)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Allocate a DML_BUFFER_TENSOR_DESC in native memory.
     * Layout (x64): DataType(4)+Flags(4)+DimCount(4)+pad(4)+Sizes(8)+Strides(8)+TotalSize(8)+Alignment(4)+pad(4) = 48 bytes
     */
    public static MemorySegment allocBufferTensorDesc(Arena arena, int dataType, int[] sizes,
                                                       int[] strides, long totalSizeBytes) {
        MemorySegment desc = arena.allocate(48, 8);
        desc.set(ValueLayout.JAVA_INT, 0, dataType);
        desc.set(ValueLayout.JAVA_INT, 4, DML_TENSOR_FLAG_NONE);
        desc.set(ValueLayout.JAVA_INT, 8, sizes.length);
        // sizes array
        MemorySegment sizesPtr = arena.allocate((long) sizes.length * ValueLayout.JAVA_INT.byteSize(), 4);
        for (int i = 0; i < sizes.length; i++) sizesPtr.setAtIndex(ValueLayout.JAVA_INT, i, sizes[i]);
        desc.set(ValueLayout.ADDRESS, 16, sizesPtr);
        // strides (optional)
        if (strides != null) {
            MemorySegment stridesPtr = arena.allocate((long) strides.length * ValueLayout.JAVA_INT.byteSize(), 4);
            for (int i = 0; i < strides.length; i++) stridesPtr.setAtIndex(ValueLayout.JAVA_INT, i, strides[i]);
            desc.set(ValueLayout.ADDRESS, 24, stridesPtr);
        } else {
            desc.set(ValueLayout.ADDRESS, 24, MemorySegment.NULL);
        }
        desc.set(ValueLayout.JAVA_LONG, 32, totalSizeBytes);
        desc.set(ValueLayout.JAVA_INT, 40, 0); // GuaranteedBaseOffsetAlignment
        return desc;
    }

    /**
     * Allocate a DML_TENSOR_DESC pointing to a buffer tensor desc.
     * Layout: Type(4)+pad(4)+Desc(8) = 16 bytes
     */
    public static MemorySegment allocTensorDesc(Arena arena, MemorySegment bufferTensorDesc) {
        MemorySegment td = arena.allocate(16, 8);
        td.set(ValueLayout.JAVA_INT, 0, DML_TENSOR_TYPE_BUFFER);
        td.set(ValueLayout.ADDRESS, 8, bufferTensorDesc);
        return td;
    }

    /**
     * Allocate a DML_OPERATOR_DESC.
     * Layout: Type(4)+pad(4)+Desc(8) = 16 bytes
     */
    public static MemorySegment allocOperatorDesc(Arena arena, int type, MemorySegment innerDesc) {
        MemorySegment od = arena.allocate(16, 8);
        od.set(ValueLayout.JAVA_INT, 0, type);
        od.set(ValueLayout.ADDRESS, 8, innerDesc);
        return od;
    }

    /**
     * Build a DML_BINDING_TABLE_DESC.
     * Layout: Dispatchable(8)+CPUHandle(8)+GPUHandle(8)+SizeInDesc(4)+pad(4) = 32 bytes
     */
    public static MemorySegment allocBindingTableDesc(Arena arena, MemorySegment dispatchable,
                                                       long cpuHandle, long gpuHandle, int sizeInDesc) {
        MemorySegment desc = arena.allocate(32, 8);
        desc.set(ValueLayout.ADDRESS, 0, dispatchable);
        desc.set(ValueLayout.JAVA_LONG, 8, cpuHandle);
        desc.set(ValueLayout.JAVA_LONG, 16, gpuHandle);
        desc.set(ValueLayout.JAVA_INT, 24, sizeInDesc);
        return desc;
    }

    /**
     * Build a DML_BUFFER_BINDING.
     * Layout: Buffer(8)+Offset(8)+SizeInBytes(8) = 24 bytes
     */
    public static MemorySegment allocBufferBinding(Arena arena, MemorySegment buffer,
                                                    long offset, long sizeInBytes) {
        MemorySegment bb = arena.allocate(24, 8);
        bb.set(ValueLayout.ADDRESS, 0, buffer);
        bb.set(ValueLayout.JAVA_LONG, 8, offset);
        bb.set(ValueLayout.JAVA_LONG, 16, sizeInBytes);
        return bb;
    }

    /**
     * Build a DML_BINDING_DESC.
     * Layout: Type(4)+pad(4)+Desc(8) = 16 bytes
     */
    public static MemorySegment allocBindingDesc(Arena arena, int type, MemorySegment innerDesc) {
        MemorySegment bd = arena.allocate(16, 8);
        bd.set(ValueLayout.JAVA_INT, 0, type);
        bd.set(ValueLayout.ADDRESS, 8, innerDesc);
        return bd;
    }

    /** Build a "none" binding desc (no resource bound). */
    public static MemorySegment allocNoneBindingDesc(Arena arena) {
        return allocBindingDesc(arena, DML_BINDING_TYPE_NONE, MemorySegment.NULL);
    }

    /** Compute the aligned byte size for a float tensor with given element count. DML requires 4-byte aligned sizes. */
    public static long tensorByteSize(int elementCount) {
        return (long) elementCount * Float.BYTES;
    }
}
