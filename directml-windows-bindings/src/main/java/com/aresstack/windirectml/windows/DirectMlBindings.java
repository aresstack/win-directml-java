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

    private DirectMlBindings() {
    }

    // ── DML constants ────────────────────────────────────────────────────
    public static final int DML_CREATE_DEVICE_FLAG_NONE = 0;
    public static final int DML_CREATE_DEVICE_FLAG_DEBUG = 1;
    public static final int DML_EXECUTION_FLAG_NONE = 0;

    // DML_TENSOR_DATA_TYPE
    public static final int DML_TENSOR_DATA_TYPE_FLOAT32 = 1;

    // DML_TENSOR_TYPE
    public static final int DML_TENSOR_TYPE_INVALID = 0;
    public static final int DML_TENSOR_TYPE_BUFFER = 1;

    // DML_TENSOR_FLAGS
    public static final int DML_TENSOR_FLAG_NONE = 0;

    // DML_OPERATOR_TYPE values (from DirectML.h – Windows SDK 10.0.26100.0).
    // Each ID below has been verified by counting the canonical enum in
    // %WindowsSdkDir%/Include/10.0.26100.0/um/DirectML.h. Do NOT guess –
    // a wrong ID makes IDMLDevice::CreateOperator fail with E_INVALIDARG
    // because DML interprets our desc with the wrong struct layout
    // (see Git history of LayerNorm: MVN0 was 39, which is LEAKY_RELU).
    public static final int DML_OPERATOR_ELEMENT_WISE_IDENTITY = 1;
    public static final int DML_OPERATOR_ELEMENT_WISE_ADD = 4;
    /**
     * Element-wise binary divide ({@code y = a / b}) with broadcast support
     * via zero strides. FL 1.0 baseline, present in every shipping
     * {@code DirectML.dll}. Used by {@code DirectMlL2NormalizeKernel} to
     * divide a per-vector input by its scalar L2-norm broadcast over the
     * hidden axis.
     */
    public static final int DML_OPERATOR_ELEMENT_WISE_DIVIDE = 10;
    public static final int DML_OPERATOR_ELEMENT_WISE_MULTIPLY = 24;
    /**
     * Element-wise unary square-root ({@code y = sqrt(scale·x + bias)})
     * with an optional {@code DML_SCALE_BIAS}. FL 1.0 baseline. Used by
     * {@code DirectMlL2NormalizeKernel} to fold {@code sqrt(s² + ε²)}
     * into a single dispatch.
     */
    public static final int DML_OPERATOR_ELEMENT_WISE_SQRT = 29;
    public static final int DML_OPERATOR_ACTIVATION_RELU = 44;
    /**
     * Softmax over the innermost dimension (FL 2.0). Applied to the last
     * tensor axis: {@code y_i = exp(x_i) / sum_j exp(x_j)}. Used by the
     * attention kernel for the score normalisation.
     */
    public static final int DML_OPERATOR_ACTIVATION_SOFTMAX = 48;
    public static final int DML_OPERATOR_CONVOLUTION = 53;
    public static final int DML_OPERATOR_GEMM = 54;
    public static final int DML_OPERATOR_MAX_POOLING = 58;
    public static final int DML_OPERATOR_BATCH_NORMALIZATION = 72;
    public static final int DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION = 73;
    /**
     * ERF primitive ({@code y = erf(scale·x + bias)}) with optional
     * {@code DML_SCALE_BIAS} pre-processing. FL 2.0 baseline – present
     * in every {@code DirectML.dll} shipped with Windows 10/11.
     * <p>
     * Currently unused at runtime. Reserved as the building block for a
     * planned composite GELU fallback ({@code 0.5·x·(1+erf(x/√2))} via
     * ERF + IDENTITY(scale=0.5, bias=0.5) + MULTIPLY) that would let the
     * GELU kernel run on FL 5.0 in-box DLLs such as Windows 11 RTM 1.8.0,
     * where {@link #DML_OPERATOR_ACTIVATION_GELU} is unavailable.
     */
    public static final int DML_OPERATOR_ELEMENT_WISE_ERF = 81;
    /**
     * MVN1 (DML_TARGET_VERSION ≥ 0x2100). Verified by counting the enum:
     * sits between SPACE_TO_DEPTH1 and RESAMPLE1 → 115. Not used by the
     * LayerNorm kernel (we use MVN0=73), kept for completeness.
     */
    public static final int DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1 = 115;
    /**
     * Native exact GELU ({@code 0.5·x·(1+erf(x/√2))}). Requires
     * DML_FEATURE_LEVEL_5_1, i.e. an in-box {@code DirectML.dll} ≥ 1.10
     * (Windows 11 22H2-Update, May 2022). Modern Windows 11 builds ship
     * 1.15.5+ which covers this op. Used directly by
     * {@code DirectMlGeluKernel}; on hosts whose in-box DLL is older,
     * point {@link #SYS_PROP_DIRECTML_DLL} at a bundled redistributable.
     */
    public static final int DML_OPERATOR_ACTIVATION_GELU = 157;
    /**
     * Multi-head attention fused op (DML_TARGET_VERSION ≥ 0x6100). Reserved
     * for the upcoming {@code DirectMlAttentionKernel} sprint.
     */
    public static final int DML_OPERATOR_MULTIHEAD_ATTENTION = 164;

    // DML_CONVOLUTION_MODE / DIRECTION
    public static final int DML_CONVOLUTION_MODE_CROSS_CORRELATION = 0;
    public static final int DML_CONVOLUTION_DIRECTION_FORWARD = 0;

    // DML_MATRIX_TRANSFORM
    public static final int DML_MATRIX_TRANSFORM_NONE = 0;
    public static final int DML_MATRIX_TRANSFORM_TRANSPOSE = 1;

    // DML_BINDING_TYPE
    public static final int DML_BINDING_TYPE_NONE = 0;
    public static final int DML_BINDING_TYPE_BUFFER = 1;

    // ── DML_FEATURE / DML_FEATURE_LEVEL ──────────────────────────────────
    /**
     * DML_FEATURE enum value for querying supported feature levels.
     */
    public static final int DML_FEATURE_FEATURE_LEVELS = 1;

    // DML_FEATURE_LEVEL enum (UINT). Values from DirectML.h.
    public static final int DML_FEATURE_LEVEL_1_0 = 0x1000;
    public static final int DML_FEATURE_LEVEL_2_0 = 0x2000;
    public static final int DML_FEATURE_LEVEL_2_1 = 0x2100;
    public static final int DML_FEATURE_LEVEL_3_0 = 0x3000;
    public static final int DML_FEATURE_LEVEL_3_1 = 0x3100;
    public static final int DML_FEATURE_LEVEL_4_0 = 0x4000;
    public static final int DML_FEATURE_LEVEL_4_1 = 0x4100;
    public static final int DML_FEATURE_LEVEL_5_0 = 0x5000;
    /**
     * Introduces native {@link #DML_OPERATOR_ACTIVATION_GELU} (op 157).
     */
    public static final int DML_FEATURE_LEVEL_5_1 = 0x5100;
    public static final int DML_FEATURE_LEVEL_5_2 = 0x5200;
    public static final int DML_FEATURE_LEVEL_6_0 = 0x6000;
    /**
     * Introduces native {@link #DML_OPERATOR_MULTIHEAD_ATTENTION} (op 164).
     */
    public static final int DML_FEATURE_LEVEL_6_1 = 0x6100;
    public static final int DML_FEATURE_LEVEL_6_2 = 0x6200;
    public static final int DML_FEATURE_LEVEL_6_3 = 0x6300;
    public static final int DML_FEATURE_LEVEL_6_4 = 0x6400;

    /**
     * All known DML feature levels in ascending order – used for the max-FL query.
     */
    private static final int[] ALL_FEATURE_LEVELS = {
            DML_FEATURE_LEVEL_1_0, DML_FEATURE_LEVEL_2_0, DML_FEATURE_LEVEL_2_1,
            DML_FEATURE_LEVEL_3_0, DML_FEATURE_LEVEL_3_1,
            DML_FEATURE_LEVEL_4_0, DML_FEATURE_LEVEL_4_1,
            DML_FEATURE_LEVEL_5_0, DML_FEATURE_LEVEL_5_1, DML_FEATURE_LEVEL_5_2,
            DML_FEATURE_LEVEL_6_0, DML_FEATURE_LEVEL_6_1, DML_FEATURE_LEVEL_6_2,
            DML_FEATURE_LEVEL_6_3, DML_FEATURE_LEVEL_6_4
    };

    // ── IDMLDevice vtable slots ──────────────────────────────────────────
    // IUnknown: 0-2, IDMLObject: 3-6
    static final int DML_DEV_CHECK_FEATURE_SUPPORT = 7;
    static final int DML_DEV_CREATE_OPERATOR = 8;
    static final int DML_DEV_COMPILE_OPERATOR = 9;
    static final int DML_DEV_CREATE_OPERATOR_INITIALIZER = 10;
    static final int DML_DEV_CREATE_COMMAND_RECORDER = 11;
    static final int DML_DEV_CREATE_BINDING_TABLE = 12;

    // IDMLDispatchable: GetBindingProperties = slot 8
    // Inheritance: IUnknown(0-2) → IDMLObject(3-6) → IDMLDeviceChild::GetDevice(7) → IDMLPageable(none) → IDMLDispatchable(8)
    static final int DISPATCHABLE_GET_BINDING_PROPERTIES = 8;

    // IDMLCommandRecorder: RecordDispatch = slot 8
    // Inheritance: IUnknown(0-2) → IDMLObject(3-6) → IDMLDeviceChild::GetDevice(7) → IDMLPageable(none) → IDMLCommandRecorder(8)
    static final int RECORDER_RECORD_DISPATCH = 8;

    // IDMLBindingTable inherits from IDMLDeviceChild (NOT just IDMLObject!)
    // Inheritance: IUnknown(0-2) → IDMLObject(3-6) → IDMLDeviceChild::GetDevice(7) → IDMLBindingTable(8+)
    static final int BT_BIND_INPUTS = 8;
    static final int BT_BIND_OUTPUTS = 9;
    static final int BT_BIND_TEMPORARY = 10;
    static final int BT_BIND_PERSISTENT = 11;

    // ── DMLCreateDevice function handle ──────────────────────────────────
    private static final FunctionDescriptor DML_CREATE_DEVICE_DESC =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);

    /**
     * System property name for an optional absolute path to a
     * {@code DirectML.dll}. When unset (the default), the binding loads the
     * in-box {@code C:\Windows\System32\DirectML.dll} via the standard
     * Windows DLL search order.
     * <p>
     * Use this to point at a Microsoft.AI.DirectML redistributable that you
     * ship next to your application (e.g. {@code app/native/DirectML.dll}).
     * The classic dev-machine in-box DLL ships as version 1.8.0 (May 2022)
     * which predates GELU (FL 5.1). On a modern Windows 11 build the in-box
     * DLL is 1.15.5+ which has every operator we currently use.
     * <p>
     * <b>Policy:</b> only forward-compatible redistributables that the
     * <em>application</em> bundles deliberately should be referenced here.
     * Never point this at a DLL that some other application installed in
     * an arbitrary location (e.g. {@code C:\Program Files\WSL\...}).
     */
    public static final String SYS_PROP_DIRECTML_DLL = "windirectml.directml.dll";

    private static volatile MethodHandle dmlCreateDeviceHandle;
    private static volatile SymbolLookup dmlSymbolLookup;
    private static volatile String dmlSourceLabel; // for logging

    /**
     * Resolve the {@code DirectML.dll} {@link SymbolLookup} once. Honours
     * the {@link #SYS_PROP_DIRECTML_DLL} system property when set, falls
     * back to the in-box system DLL otherwise.
     */
    private static SymbolLookup getDmlSymbolLookup() {
        if (dmlSymbolLookup == null) {
            synchronized (DirectMlBindings.class) {
                if (dmlSymbolLookup == null) {
                    String override = System.getProperty(SYS_PROP_DIRECTML_DLL);
                    if (override != null && !override.isBlank()) {
                        java.nio.file.Path p = java.nio.file.Path.of(override.trim());
                        log.info("Loading DirectML from -D{}={}", SYS_PROP_DIRECTML_DLL, p);
                        dmlSymbolLookup = SymbolLookup.libraryLookup(p, Arena.global());
                        dmlSourceLabel = p.toString();
                    } else {
                        log.info("Loading in-box DirectML.dll (System32) via default DLL search order");
                        dmlSymbolLookup = SymbolLookup.libraryLookup("DirectML.dll", Arena.global());
                        dmlSourceLabel = "DirectML.dll (system)";
                    }
                }
            }
        }
        return dmlSymbolLookup;
    }

    /**
     * @return human-readable description of where DirectML.dll was loaded from.
     */
    public static String directMlSource() {
        getDmlSymbolLookup();
        return dmlSourceLabel;
    }

    private static MethodHandle getDmlCreateDeviceHandle() {
        if (dmlCreateDeviceHandle == null) {
            synchronized (DirectMlBindings.class) {
                if (dmlCreateDeviceHandle == null) {
                    SymbolLookup dml = getDmlSymbolLookup();
                    MemorySegment addr = dml.find("DMLCreateDevice")
                            .orElseThrow(() -> new UnsatisfiedLinkError("DMLCreateDevice not found in " + dmlSourceLabel));
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
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("DMLCreateDevice failed", t);
        }
    }

    public static boolean isAvailable() {
        try {
            getDmlCreateDeviceHandle();
            return true;
        } catch (Exception | UnsatisfiedLinkError e) {
            return false;
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Operator creation & compilation
    // ══════════════════════════════════════════════════════════════════════

    /**
     * IDMLDevice::CreateOperator (vtable slot 8).
     *
     * @param dmlDevice IDMLDevice COM pointer
     * @param opDesc    pointer to filled DML_OPERATOR_DESC struct
     * @param arena     arena for allocations
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
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateOperator failed", t);
        }
    }

    /**
     * IDMLDevice::CompileOperator (vtable slot 9).
     *
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
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CompileOperator failed", t);
        }
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
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateOperatorInitializer failed", t);
        }
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
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateCommandRecorder failed", t);
        }
    }

    /**
     * IDMLDevice::CreateBindingTable (vtable slot 12).
     *
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
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateBindingTable failed", t);
        }
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
            return new long[]{descCount, tempSize, persistSize};
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

    /**
     * IDMLBindingTable::BindInputs (slot 8).
     */
    public static void bindInputs(MemorySegment bt, int count, MemorySegment bindings) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, BT_BIND_INPUTS,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            mh.invokeExact(bt, count, bindings);
        } catch (Throwable t) {
            throw new RuntimeException("BindInputs failed", t);
        }
    }

    /**
     * IDMLBindingTable::BindOutputs (slot 9).
     */
    public static void bindOutputs(MemorySegment bt, int count, MemorySegment bindings) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, BT_BIND_OUTPUTS,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            mh.invokeExact(bt, count, bindings);
        } catch (Throwable t) {
            throw new RuntimeException("BindOutputs failed", t);
        }
    }

    /**
     * IDMLBindingTable::BindTemporaryResource (slot 10).
     */
    public static void bindTemporaryResource(MemorySegment bt, MemorySegment binding) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, BT_BIND_TEMPORARY,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            mh.invokeExact(bt, binding);
        } catch (Throwable t) {
            throw new RuntimeException("BindTemporary failed", t);
        }
    }

    /**
     * IDMLBindingTable::BindPersistentResource (slot 11).
     */
    public static void bindPersistentResource(MemorySegment bt, MemorySegment binding) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, BT_BIND_PERSISTENT,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            mh.invokeExact(bt, binding);
        } catch (Throwable t) {
            throw new RuntimeException("BindPersistent failed", t);
        }
    }

    /**
     * IDMLBindingTable::Reset (slot 12).
     */
    public static void resetBindingTable(MemorySegment bt, MemorySegment newDesc) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(bt, 12,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(bt, newDesc);
            HResult.check(hr, "IDMLBindingTable::Reset");
        } catch (Throwable t) {
            throw new RuntimeException("BindingTable::Reset failed", t);
        }
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

    /**
     * Build a "none" binding desc (no resource bound).
     */
    public static MemorySegment allocNoneBindingDesc(Arena arena) {
        return allocBindingDesc(arena, DML_BINDING_TYPE_NONE, MemorySegment.NULL);
    }

    /**
     * Compute the aligned byte size for a float tensor with given element count. DML requires 4-byte aligned sizes.
     */
    public static long tensorByteSize(int elementCount) {
        return (long) elementCount * Float.BYTES;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Feature-level detection
    // ══════════════════════════════════════════════════════════════════════

    /**
     * IDMLDevice::CheckFeatureSupport (vtable slot 7).
     * <p>
     * Native signature:
     * <pre>{@code
     * HRESULT CheckFeatureSupport(
     *     DML_FEATURE feature,
     *     UINT featureQueryDataSize,
     *     const void* featureQueryData,
     *     UINT featureSupportDataSize,
     *     void* featureSupportData);
     * }</pre>
     *
     * @param dmlDevice       IDMLDevice
     * @param feature         DML_FEATURE enum value
     * @param queryData       pointer to feature-specific query struct (may be NULL)
     * @param queryDataSize   size of the query struct
     * @param supportData     pointer to caller-allocated output struct
     * @param supportDataSize size of the output struct
     */
    public static void checkFeatureSupport(MemorySegment dmlDevice, int feature,
                                           MemorySegment queryData, int queryDataSize,
                                           MemorySegment supportData, int supportDataSize)
            throws WindowsNativeException {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(dmlDevice, DML_DEV_CHECK_FEATURE_SUPPORT,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(dmlDevice, feature,
                    queryDataSize, queryData,
                    supportDataSize, supportData);
            HResult.check(hr, "IDMLDevice::CheckFeatureSupport");
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CheckFeatureSupport failed", t);
        }
    }

    /**
     * Query the maximum DML feature level supported by {@code dmlDevice}
     * (using {@link #ALL_FEATURE_LEVELS} as the requested set). Returns
     * the raw {@code DML_FEATURE_LEVEL} UINT value, e.g. {@code 0x5100}
     * for FL 5.1.
     */
    public static int queryMaxFeatureLevel(MemorySegment dmlDevice, Arena arena)
            throws WindowsNativeException {
        // DML_FEATURE_QUERY_FEATURE_LEVELS:
        //   UINT  RequestedFeatureLevelCount       (offset 0, 4)
        //   pad                                    (offset 4, 4)
        //   const DML_FEATURE_LEVEL* RequestedFeatureLevels  (offset 8, 8)
        // → 16 bytes
        MemorySegment levelsArr = arena.allocate((long) ALL_FEATURE_LEVELS.length * Integer.BYTES, 4);
        for (int i = 0; i < ALL_FEATURE_LEVELS.length; i++) {
            levelsArr.setAtIndex(ValueLayout.JAVA_INT, i, ALL_FEATURE_LEVELS[i]);
        }
        MemorySegment query = arena.allocate(16, 8);
        query.set(ValueLayout.JAVA_INT, 0, ALL_FEATURE_LEVELS.length);
        query.set(ValueLayout.ADDRESS, 8, levelsArr);

        // DML_FEATURE_DATA_FEATURE_LEVELS: single DML_FEATURE_LEVEL (UINT) = 4 bytes
        MemorySegment out = arena.allocate(4, 4);
        checkFeatureSupport(dmlDevice, DML_FEATURE_FEATURE_LEVELS,
                query, 16, out, 4);
        return out.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Human-readable {@code "5.1"} / {@code "6.1"} formatting of a raw
     * {@code DML_FEATURE_LEVEL} value such as {@code 0x5100}.
     */
    public static String formatFeatureLevel(int rawLevel) {
        int major = (rawLevel >>> 12) & 0xF;
        int minor = (rawLevel >>> 8) & 0xF;
        return major + "." + minor;
    }

    /**
     * @return {@code true} iff {@link #DML_OPERATOR_ACTIVATION_GELU} is available natively.
     */
    public static boolean supportsFusedGelu(int featureLevel) {
        return Integer.compareUnsigned(featureLevel, DML_FEATURE_LEVEL_5_1) >= 0;
    }

    /**
     * @return {@code true} iff {@link #DML_OPERATOR_MULTIHEAD_ATTENTION} is available.
     */
    public static boolean supportsMultiHeadAttention(int featureLevel) {
        return Integer.compareUnsigned(featureLevel, DML_FEATURE_LEVEL_6_1) >= 0;
    }
}
