package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Java 21 FFM bindings for {@code d3d12.dll} – Direct3D 12 device and resource management.
 * <p>
 * Calls directly into the Windows system DLL. No third-party wrapper.
 */
public final class D3D12Bindings {

    private static final Logger log = LoggerFactory.getLogger(D3D12Bindings.class);

    private D3D12Bindings() {
    }

    // ── D3D_FEATURE_LEVEL constants ──────────────────────────────────────
    public static final int D3D_FEATURE_LEVEL_11_0 = 0xb000;
    public static final int D3D_FEATURE_LEVEL_11_1 = 0xb100;
    public static final int D3D_FEATURE_LEVEL_12_0 = 0xc000;
    public static final int D3D_FEATURE_LEVEL_12_1 = 0xc100;

    // ── D3D12 constants ──────────────────────────────────────────────────
    public static final int D3D12_HEAP_TYPE_DEFAULT = 1;
    public static final int D3D12_HEAP_TYPE_UPLOAD = 2;
    public static final int D3D12_HEAP_TYPE_READBACK = 3;

    public static final int D3D12_HEAP_FLAG_NONE = 0;

    public static final int D3D12_RESOURCE_DIMENSION_BUFFER = 1;

    public static final int D3D12_RESOURCE_STATE_COMMON = 0;
    public static final int D3D12_RESOURCE_STATE_UNORDERED_ACCESS = 0x8;
    public static final int D3D12_RESOURCE_STATE_COPY_DEST = 0x400;
    public static final int D3D12_RESOURCE_STATE_COPY_SOURCE = 0x800;
    public static final int D3D12_RESOURCE_STATE_GENERIC_READ = 0x1 | 0x2 | 0x40 | 0x80 | 0x200 | 0x800;

    public static final int D3D12_RESOURCE_FLAG_NONE = 0;
    public static final int D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS = 0x4;

    public static final int D3D12_COMMAND_LIST_TYPE_DIRECT = 0;
    public static final int D3D12_COMMAND_LIST_TYPE_COMPUTE = 2;

    public static final int D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV = 0;
    public static final int D3D12_DESCRIPTOR_HEAP_FLAG_NONE = 0;
    public static final int D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE = 1;

    public static final int D3D12_FENCE_FLAG_NONE = 0;

    public static final int D3D12_RESOURCE_BARRIER_TYPE_TRANSITION = 0;
    public static final int D3D12_RESOURCE_BARRIER_TYPE_UAV = 2;
    public static final int D3D12_RESOURCE_BARRIER_FLAG_NONE = 0;
    public static final int D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES = 0xFFFFFFFF;

    public static final int DXGI_FORMAT_UNKNOWN = 0;

    public static final int D3D12_QUERY_HEAP_TYPE_TIMESTAMP = 1;
    public static final int D3D12_QUERY_TYPE_TIMESTAMP = 2;

    public static final int D3D12_TEXTURE_LAYOUT_ROW_MAJOR = 1;

    // ── ID3D12Device vtable slot indices ─────────────────────────────────
    // IUnknown: 0-2, ID3D12Object: 3-6, ID3D12Device starts at 7
    static final int DEV_CREATE_COMMAND_QUEUE = 8;
    static final int DEV_CREATE_COMMAND_ALLOCATOR = 9;
    static final int DEV_CREATE_COMMAND_LIST = 12;
    static final int DEV_CREATE_DESCRIPTOR_HEAP = 14;
    static final int DEV_GET_DESCRIPTOR_INCREMENT = 15;
    static final int DEV_CREATE_COMMITTED_RESOURCE = 27;
    static final int DEV_CREATE_FENCE = 36;
    static final int DEV_CREATE_QUERY_HEAP = 39;

    // ── Function descriptor for D3D12CreateDevice ────────────────────────
    private static final FunctionDescriptor D3D12_CREATE_DEVICE_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,   // HRESULT
                    ValueLayout.ADDRESS,     // IUnknown *pAdapter
                    ValueLayout.JAVA_INT,   // D3D_FEATURE_LEVEL
                    ValueLayout.ADDRESS,     // REFIID
                    ValueLayout.ADDRESS      // void **ppDevice
            );

    private static volatile MethodHandle d3d12CreateDeviceHandle;

    private static MethodHandle getD3D12CreateDeviceHandle() {
        if (d3d12CreateDeviceHandle == null) {
            synchronized (D3D12Bindings.class) {
                if (d3d12CreateDeviceHandle == null) {
                    SymbolLookup d3d12 = SymbolLookup.libraryLookup("d3d12.dll", Arena.global());
                    MemorySegment addr = d3d12.find("D3D12CreateDevice")
                            .orElseThrow(() -> new UnsatisfiedLinkError("D3D12CreateDevice not found in d3d12.dll"));
                    d3d12CreateDeviceHandle = Linker.nativeLinker()
                            .downcallHandle(addr, D3D12_CREATE_DEVICE_DESC);
                    log.debug("Resolved D3D12CreateDevice at {}", addr);
                }
            }
        }
        return d3d12CreateDeviceHandle;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Device creation
    // ══════════════════════════════════════════════════════════════════════

    public static MemorySegment createDevice(MemorySegment adapter, int featureLevel, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12Device_BYTES);
            MemorySegment ppDevice = arena.allocate(ValueLayout.ADDRESS);
            int hr = (int) getD3D12CreateDeviceHandle().invokeExact(adapter, featureLevel, riid, ppDevice);
            HResult.check(hr, "D3D12CreateDevice");
            MemorySegment device = ppDevice.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
            log.info("ID3D12Device created: {} (featureLevel=0x{})", device, Integer.toHexString(featureLevel));
            return device;
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("D3D12CreateDevice invocation failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Debug layer (D3D12GetDebugInterface → ID3D12Debug::EnableDebugLayer)
    // ══════════════════════════════════════════════════════════════════════

    private static final FunctionDescriptor D3D12_GET_DEBUG_INTERFACE_DESC =
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);
    private static volatile MethodHandle d3d12GetDebugInterfaceHandle;

    private static MethodHandle getD3D12GetDebugInterfaceHandle() {
        if (d3d12GetDebugInterfaceHandle == null) {
            synchronized (D3D12Bindings.class) {
                if (d3d12GetDebugInterfaceHandle == null) {
                    SymbolLookup d3d12 = SymbolLookup.libraryLookup("d3d12.dll", Arena.global());
                    MemorySegment addr = d3d12.find("D3D12GetDebugInterface")
                            .orElseThrow(() -> new UnsatisfiedLinkError("D3D12GetDebugInterface not in d3d12.dll"));
                    d3d12GetDebugInterfaceHandle = Linker.nativeLinker()
                            .downcallHandle(addr, D3D12_GET_DEBUG_INTERFACE_DESC);
                }
            }
        }
        return d3d12GetDebugInterfaceHandle;
    }

    /**
     * Activate the D3D12 validation layer <i>before</i> any device is created.
     * Best-effort – if the optional "Graphics Tools" feature is missing on the
     * machine, this logs a warning and returns {@code false} instead of throwing.
     *
     * @return {@code true} if the debug layer was enabled successfully
     */
    public static boolean enableDebugLayer(Arena arena) {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12Debug_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            int hr = (int) getD3D12GetDebugInterfaceHandle().invokeExact(riid, pp);
            if (hr != 0) {
                log.warn("D3D12GetDebugInterface failed: HRESULT 0x{} – continuing without debug layer",
                        Integer.toHexString(hr));
                return false;
            }
            MemorySegment dbg = pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);

            // ID3D12Debug::EnableDebugLayer (vtable slot 3, void return, this only).
            MethodHandle enable = DxgiBindings.vtableMethod(dbg, 3,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
            enable.invokeExact(dbg);
            DxgiBindings.release(dbg);
            log.info("D3D12 debug layer enabled");
            return true;
        } catch (Throwable t) {
            log.warn("Could not enable D3D12 debug layer: {} – is the 'Graphics Tools' optional "
                    + "feature installed?", t.getMessage());
            return false;
        }
    }

    /**
     * Query {@code ID3D12InfoQueue} from the given device. Returns {@code null}
     * if the device has no info queue (debug layer not enabled, or device
     * created without {@code D3D12_CREATE_DEVICE_FLAG_DEBUG}).
     */
    public static MemorySegment queryInfoQueue(MemorySegment d3d12Device, Arena arena) {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12InfoQueue_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            // ID3D12Device inherits IUnknown::QueryInterface at vtable slot 0.
            MethodHandle qi = DxgiBindings.vtableMethod(d3d12Device, 0,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) qi.invokeExact(d3d12Device, riid, pp);
            if (hr != 0) return null;
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (Throwable t) {
            log.debug("queryInfoQueue failed: {}", t.getMessage());
            return null;
        }
    }

    /**
     * Drain all currently stored messages from an {@code ID3D12InfoQueue}.
     * <p>
     * Returns a human-readable list of message texts (severity + description).
     * Clears the queue afterwards. Best-effort: any internal failure returns
     * an empty list.
     *
     * @param infoQueue handle returned by {@link #queryInfoQueue}, may be {@code null}
     */
    public static java.util.List<String> drainInfoQueue(MemorySegment infoQueue, Arena arena) {
        java.util.List<String> out = new java.util.ArrayList<>();
        if (infoQueue == null) return out;
        try {
            // ID3D12InfoQueue vtable (post-IUnknown):
            //   3 SetMessageCountLimit
            //   4 ClearStoredMessages
            //   5 GetMessage(this, idx, pMessage, pMessageByteLength) -> HRESULT
            //   6..7 GetNumMessagesAllowed/Denied
            //   8 GetNumStoredMessages() -> UINT64
            MethodHandle getNumStored = DxgiBindings.vtableMethod(infoQueue, 8,
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            MethodHandle getMessage = DxgiBindings.vtableMethod(infoQueue, 5,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            MethodHandle clear = DxgiBindings.vtableMethod(infoQueue, 4,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

            long n = (long) getNumStored.invokeExact(infoQueue);
            for (long i = 0; i < n; i++) {
                // First call: pMessage=NULL → driver writes required byte length into pSize.
                MemorySegment pSize = arena.allocate(ValueLayout.JAVA_LONG);
                int hr = (int) getMessage.invokeExact(infoQueue, i, MemorySegment.NULL, pSize);
                if (hr != 0) continue;
                long sz = pSize.get(ValueLayout.JAVA_LONG, 0);
                if (sz <= 32) continue;
                MemorySegment buf = arena.allocate(sz, 8);
                hr = (int) getMessage.invokeExact(infoQueue, i, buf, pSize);
                if (hr != 0) continue;
                // D3D12_MESSAGE layout:
                //   Category(4) + Severity(4) + ID(4) + pad(4) + pDescription(8) + DescByteLength(8) = 32
                int severity = buf.get(ValueLayout.JAVA_INT, 4);
                MemorySegment descPtr = buf.get(ValueLayout.ADDRESS, 16);
                long descLen = buf.get(ValueLayout.JAVA_LONG, 24);
                String txt;
                if (descPtr.equals(MemorySegment.NULL) || descLen <= 0) {
                    txt = "(no description)";
                } else {
                    MemorySegment desc = descPtr.reinterpret(descLen);
                    byte[] bytes = new byte[(int) Math.min(descLen, 4096)];
                    MemorySegment.copy(desc, ValueLayout.JAVA_BYTE, 0, bytes, 0, bytes.length);
                    int end = bytes.length;
                    while (end > 0 && bytes[end - 1] == 0) end--;
                    txt = new String(bytes, 0, end, java.nio.charset.StandardCharsets.US_ASCII);
                }
                out.add("[sev=" + severity + "] " + txt);
            }
            clear.invokeExact(infoQueue);
        } catch (Throwable t) {
            log.debug("drainInfoQueue failed: {}", t.getMessage());
        }
        return out;
    }


    // ══════════════════════════════════════════════════════════════════════
    // Command queue (existing)
    // ══════════════════════════════════════════════════════════════════════

    public static MemorySegment createCommandQueue(MemorySegment device, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment desc = arena.allocate(16, 4);
            desc.set(ValueLayout.JAVA_INT, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);
            desc.set(ValueLayout.JAVA_INT, 4, 0);
            desc.set(ValueLayout.JAVA_INT, 8, 0);
            desc.set(ValueLayout.JAVA_INT, 12, 0);
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12CommandQueue_BYTES);
            MemorySegment ppQueue = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle createCQ = DxgiBindings.vtableMethod(device, DEV_CREATE_COMMAND_QUEUE,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) createCQ.invokeExact(device, desc, riid, ppQueue);
            HResult.check(hr, "ID3D12Device::CreateCommandQueue");
            MemorySegment queue = ppQueue.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
            log.info("ID3D12CommandQueue created: {}", queue);
            return queue;
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateCommandQueue invocation failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Command allocator
    // ══════════════════════════════════════════════════════════════════════

    public static MemorySegment createCommandAllocator(MemorySegment device, int type, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12CommandAllocator_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(device, DEV_CREATE_COMMAND_ALLOCATOR,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(device, type, riid, pp);
            HResult.check(hr, "ID3D12Device::CreateCommandAllocator");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateCommandAllocator failed", t);
        }
    }

    /**
     * ID3D12CommandAllocator::Reset (vtable slot 8 on ID3D12CommandAllocator).
     */
    public static void resetCommandAllocator(MemorySegment allocator) throws WindowsNativeException {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(allocator, 8,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(allocator);
            HResult.check(hr, "ID3D12CommandAllocator::Reset");
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CommandAllocator::Reset failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Graphics command list
    // ══════════════════════════════════════════════════════════════════════

    /**
     * ID3D12Device::CreateCommandList.
     * The command list is created in the recording state.
     */
    public static MemorySegment createCommandList(MemorySegment device, int type,
                                                  MemorySegment allocator, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12GraphicsCommandList_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(device, DEV_CREATE_COMMAND_LIST,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(device, 0, type, allocator,
                    MemorySegment.NULL, riid, pp);
            HResult.check(hr, "ID3D12Device::CreateCommandList");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateCommandList failed", t);
        }
    }

    /**
     * ID3D12GraphicsCommandList::Close (vtable slot 9).
     */
    public static void closeCommandList(MemorySegment cmdList) throws WindowsNativeException {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(cmdList, 9,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(cmdList);
            HResult.check(hr, "ID3D12GraphicsCommandList::Close");
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CommandList::Close failed", t);
        }
    }

    /**
     * ID3D12GraphicsCommandList::Reset (vtable slot 10).
     */
    public static void resetCommandList(MemorySegment cmdList, MemorySegment allocator)
            throws WindowsNativeException {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(cmdList, 10,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(cmdList, allocator, MemorySegment.NULL);
            HResult.check(hr, "ID3D12GraphicsCommandList::Reset");
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CommandList::Reset failed", t);
        }
    }

    /**
     * ID3D12GraphicsCommandList::CopyBufferRegion (vtable slot 15).
     */
    public static void copyBufferRegion(MemorySegment cmdList, MemorySegment dst, long dstOffset,
                                        MemorySegment src, long srcOffset, long numBytes) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(cmdList, 15,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            mh.invokeExact(cmdList, dst, dstOffset, src, srcOffset, numBytes);
        } catch (Throwable t) {
            log.error("CopyBufferRegion failed", t);
        }
    }

    /**
     * ID3D12GraphicsCommandList::ResourceBarrier (vtable slot 26).
     * Records a UAV barrier (no specific resource) to sync between DML dispatches.
     */
    public static void uavBarrier(MemorySegment cmdList, Arena arena) {
        try {
            // D3D12_RESOURCE_BARRIER for UAV: Type=2, Flags=0, UAV.pResource=NULL
            MemorySegment barrier = arena.allocate(32, 8); // max barrier struct size
            barrier.set(ValueLayout.JAVA_INT, 0, D3D12_RESOURCE_BARRIER_TYPE_UAV);
            barrier.set(ValueLayout.JAVA_INT, 4, D3D12_RESOURCE_BARRIER_FLAG_NONE);
            barrier.set(ValueLayout.ADDRESS, 8, MemorySegment.NULL); // UAV.pResource = NULL = all
            MethodHandle mh = DxgiBindings.vtableMethod(cmdList, 26,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            mh.invokeExact(cmdList, 1, barrier);
            WarpSubmissionStats.recordUavBarrier(); // GEMMA-WARP-14b: count recorded UAV barriers
        } catch (Throwable t) {
            log.error("ResourceBarrier (UAV) failed", t);
        }
    }

    /**
     * Record a transition barrier on a resource.
     */
    public static void transitionBarrier(MemorySegment cmdList, MemorySegment resource,
                                         int stateBefore, int stateAfter, Arena arena) {
        try {
            MemorySegment barrier = arena.allocate(32, 8);
            barrier.set(ValueLayout.JAVA_INT, 0, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION);
            barrier.set(ValueLayout.JAVA_INT, 4, D3D12_RESOURCE_BARRIER_FLAG_NONE);
            barrier.set(ValueLayout.ADDRESS, 8, resource);
            barrier.set(ValueLayout.JAVA_INT, 16, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
            barrier.set(ValueLayout.JAVA_INT, 20, stateBefore);
            barrier.set(ValueLayout.JAVA_INT, 24, stateAfter);
            MethodHandle mh = DxgiBindings.vtableMethod(cmdList, 26,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            mh.invokeExact(cmdList, 1, barrier);
        } catch (Throwable t) {
            log.error("ResourceBarrier (transition) failed", t);
        }
    }

    /**
     * ID3D12GraphicsCommandList::SetDescriptorHeaps (vtable slot 28).
     */
    public static void setDescriptorHeaps(MemorySegment cmdList, MemorySegment heap, Arena arena) {
        try {
            MemorySegment heapArray = arena.allocate(ValueLayout.ADDRESS);
            heapArray.set(ValueLayout.ADDRESS, 0, heap);
            MethodHandle mh = DxgiBindings.vtableMethod(cmdList, 28,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            mh.invokeExact(cmdList, 1, heapArray);
        } catch (Throwable t) {
            log.error("SetDescriptorHeaps failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Committed resources (GPU buffers)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Create a committed D3D12 buffer resource.
     *
     * @param device    ID3D12Device
     * @param heapType  D3D12_HEAP_TYPE_DEFAULT / UPLOAD / READBACK
     * @param sizeBytes buffer size in bytes
     * @param resFlags  D3D12_RESOURCE_FLAG_NONE or _ALLOW_UNORDERED_ACCESS
     * @param initState initial resource state
     * @param arena     arena for allocations
     * @return ID3D12Resource COM pointer
     */
    public static MemorySegment createBuffer(MemorySegment device, int heapType,
                                             long sizeBytes, int resFlags, int initState,
                                             Arena arena) throws WindowsNativeException {
        try {
            // D3D12 needs minimum buffer size; DML commonly requires >= 4 bytes
            sizeBytes = Math.max(sizeBytes, 4);

            // D3D12_HEAP_PROPERTIES: {Type(4), CPUPageProp(4), MemPoolPref(4), CreationNodeMask(4), VisibleNodeMask(4)} = 20 bytes
            MemorySegment heapProps = arena.allocate(24, 8);
            heapProps.set(ValueLayout.JAVA_INT, 0, heapType);
            // rest are 0 (default CPU page, default memory pool, node masks)

            // D3D12_RESOURCE_DESC for BUFFER: 56 bytes with alignment
            MemorySegment resDesc = arena.allocate(56, 8);
            resDesc.set(ValueLayout.JAVA_INT, 0, D3D12_RESOURCE_DIMENSION_BUFFER); // Dimension
            resDesc.set(ValueLayout.JAVA_LONG, 8, 0L);        // Alignment (default)
            resDesc.set(ValueLayout.JAVA_LONG, 16, sizeBytes); // Width = size
            resDesc.set(ValueLayout.JAVA_INT, 24, 1);          // Height = 1
            resDesc.set(ValueLayout.JAVA_SHORT, 28, (short) 1); // DepthOrArraySize = 1
            resDesc.set(ValueLayout.JAVA_SHORT, 30, (short) 1); // MipLevels = 1
            resDesc.set(ValueLayout.JAVA_INT, 32, DXGI_FORMAT_UNKNOWN); // Format
            resDesc.set(ValueLayout.JAVA_INT, 36, 1);          // SampleDesc.Count = 1
            resDesc.set(ValueLayout.JAVA_INT, 40, 0);          // SampleDesc.Quality = 0
            resDesc.set(ValueLayout.JAVA_INT, 44, D3D12_TEXTURE_LAYOUT_ROW_MAJOR); // Layout
            resDesc.set(ValueLayout.JAVA_INT, 48, resFlags);   // Flags

            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12Resource_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);

            MethodHandle mh = DxgiBindings.vtableMethod(device, DEV_CREATE_COMMITTED_RESOURCE,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(device, heapProps, D3D12_HEAP_FLAG_NONE,
                    resDesc, initState, MemorySegment.NULL, riid, pp);
            HResult.check(hr, "ID3D12Device::CreateCommittedResource");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateCommittedResource failed", t);
        }
    }

    /**
     * Create a default-heap buffer with UAV access (for DML tensor buffers).
     */
    public static MemorySegment createDefaultBuffer(MemorySegment device, long size, Arena arena)
            throws WindowsNativeException {
        return createBuffer(device, D3D12_HEAP_TYPE_DEFAULT, size,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, arena);
    }

    /**
     * Create an upload-heap buffer (CPU-writable, for uploading data to GPU).
     */
    public static MemorySegment createUploadBuffer(MemorySegment device, long size, Arena arena)
            throws WindowsNativeException {
        return createBuffer(device, D3D12_HEAP_TYPE_UPLOAD, size,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, arena);
    }

    /**
     * Create a readback-heap buffer (CPU-readable, for downloading results from GPU).
     */
    public static MemorySegment createReadbackBuffer(MemorySegment device, long size, Arena arena)
            throws WindowsNativeException {
        return createBuffer(device, D3D12_HEAP_TYPE_READBACK, size,
                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, arena);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Resource mapping (CPU read/write of upload/readback buffers)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * ID3D12Resource::Map (vtable slot 8). Returns CPU-accessible pointer.
     */
    public static MemorySegment mapResource(MemorySegment resource, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment ppData = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(resource, 8,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(resource, 0, MemorySegment.NULL, ppData);
            HResult.check(hr, "ID3D12Resource::Map");
            return ppData.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("Resource::Map failed", t);
        }
    }

    /**
     * ID3D12Resource::Unmap (vtable slot 9).
     */
    public static void unmapResource(MemorySegment resource) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(resource, 9,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            mh.invokeExact(resource, 0, MemorySegment.NULL);
        } catch (Throwable t) {
            log.warn("Resource::Unmap failed", t);
        }
    }

    /**
     * ID3D12Resource::GetGPUVirtualAddress (vtable slot 11). Returns UINT64.
     */
    public static long getGpuVirtualAddress(MemorySegment resource) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(resource, 11,
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            return (long) mh.invokeExact(resource);
        } catch (Throwable t) {
            throw new RuntimeException("GetGPUVirtualAddress failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Descriptor heap
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Create a CBV/SRV/UAV descriptor heap (shader visible).
     */
    public static MemorySegment createDescriptorHeap(MemorySegment device, int numDescriptors,
                                                     Arena arena) throws WindowsNativeException {
        try {
            // D3D12_DESCRIPTOR_HEAP_DESC: {Type(4), NumDescriptors(4), Flags(4), NodeMask(4)} = 16 bytes
            MemorySegment desc = arena.allocate(16, 4);
            desc.set(ValueLayout.JAVA_INT, 0, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            desc.set(ValueLayout.JAVA_INT, 4, numDescriptors);
            desc.set(ValueLayout.JAVA_INT, 8, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
            desc.set(ValueLayout.JAVA_INT, 12, 0); // NodeMask

            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12DescriptorHeap_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(device, DEV_CREATE_DESCRIPTOR_HEAP,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(device, desc, riid, pp);
            HResult.check(hr, "ID3D12Device::CreateDescriptorHeap");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateDescriptorHeap failed", t);
        }
    }

    /**
     * ID3D12Device::GetDescriptorHandleIncrementSize (vtable slot 15).
     */
    public static int getDescriptorIncrementSize(MemorySegment device) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(device, DEV_GET_DESCRIPTOR_INCREMENT,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            return (int) mh.invokeExact(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        } catch (Throwable t) {
            throw new RuntimeException("GetDescriptorHandleIncrementSize failed", t);
        }
    }

    /**
     * ID3D12DescriptorHeap::GetCPUDescriptorHandleForHeapStart (vtable slot 9).
     * <p>
     * On modern D3D12 (Windows 11 Agility SDK ABI), these functions take an explicit
     * output pointer parameter: GetCPUDescriptorHandleForHeapStart(this, pResult).
     * The 8-byte struct is written to *pResult instead of returned in RAX.
     */
    public static long getCpuDescriptorHandleForHeapStart(MemorySegment heap, Arena arena) {
        try {
            MemorySegment retBuf = arena.allocate(8, 8);
            // Agility SDK ABI: (this, pResult) → writes result to *pResult
            MethodHandle mh = DxgiBindings.vtableMethod(heap, 9,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            mh.invokeExact(heap, retBuf);
            long handle = retBuf.get(ValueLayout.JAVA_LONG, 0);
            log.debug("CPU descriptor handle: 0x{}", Long.toHexString(handle));
            return handle;
        } catch (Throwable t) {
            throw new RuntimeException("GetCPUDescriptorHandleForHeapStart failed", t);
        }
    }

    /**
     * ID3D12DescriptorHeap::GetGPUDescriptorHandleForHeapStart (vtable slot 10).
     * Same Agility SDK ABI as CPU handle – see above.
     */
    public static long getGpuDescriptorHandleForHeapStart(MemorySegment heap, Arena arena) {
        try {
            MemorySegment retBuf = arena.allocate(8, 8);
            MethodHandle mh = DxgiBindings.vtableMethod(heap, 10,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            mh.invokeExact(heap, retBuf);
            long handle = retBuf.get(ValueLayout.JAVA_LONG, 0);
            log.debug("GPU descriptor handle: 0x{}", Long.toHexString(handle));
            return handle;
        } catch (Throwable t) {
            throw new RuntimeException("GetGPUDescriptorHandleForHeapStart failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Timestamp query heap (GPU profiling)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Create a timestamp query heap.
     *
     * @param device     ID3D12Device
     * @param queryCount number of timestamp slots
     * @param arena      arena used for temporary descriptor memory
     * @return ID3D12QueryHeap COM pointer
     */
    public static MemorySegment createTimestampQueryHeap(MemorySegment device, int queryCount, Arena arena)
            throws WindowsNativeException {
        if (queryCount <= 0) {
            throw new IllegalArgumentException("queryCount must be positive");
        }
        try {
            // D3D12_QUERY_HEAP_DESC: Type(UINT), Count(UINT), NodeMask(UINT)
            MemorySegment desc = arena.allocate(12, 4);
            desc.set(ValueLayout.JAVA_INT, 0, D3D12_QUERY_HEAP_TYPE_TIMESTAMP);
            desc.set(ValueLayout.JAVA_INT, 4, queryCount);
            desc.set(ValueLayout.JAVA_INT, 8, 0);

            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12QueryHeap_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(device, DEV_CREATE_QUERY_HEAP,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(device, desc, riid, pp);
            HResult.check(hr, "ID3D12Device::CreateQueryHeap(TIMESTAMP)");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateQueryHeap(TIMESTAMP) failed", t);
        }
    }

    /**
     * Record a timestamp query at the current command-list position.
     */
    public static void endTimestampQuery(MemorySegment cmdList, MemorySegment queryHeap, int queryIndex) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(cmdList, 53,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            mh.invokeExact(cmdList, queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, queryIndex);
        } catch (Throwable t) {
            throw new RuntimeException("EndQuery(TIMESTAMP) failed", t);
        }
    }

    /**
     * Resolve timestamp queries into a readback resource.
     */
    public static void resolveTimestampQueries(MemorySegment cmdList, MemorySegment queryHeap,
                                               int startIndex, int queryCount,
                                               MemorySegment destinationBuffer,
                                               long destinationOffset) {
        if (queryCount <= 0) {
            return;
        }
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(cmdList, 54,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            mh.invokeExact(cmdList, queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, startIndex,
                    queryCount, destinationBuffer, destinationOffset);
        } catch (Throwable t) {
            throw new RuntimeException("ResolveQueryData(TIMESTAMP) failed", t);
        }
    }

    /**
     * Query timestamp ticks per second for the command queue.
     */
    public static long getTimestampFrequency(MemorySegment commandQueue, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment pFrequency = arena.allocate(ValueLayout.JAVA_LONG);
            MethodHandle mh = DxgiBindings.vtableMethod(commandQueue, 16,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(commandQueue, pFrequency);
            HResult.check(hr, "ID3D12CommandQueue::GetTimestampFrequency");
            long frequency = pFrequency.get(ValueLayout.JAVA_LONG, 0);
            if (frequency <= 0) {
                throw new WindowsNativeException("D3D12 timestamp frequency is not positive: " + frequency);
            }
            return frequency;
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("GetTimestampFrequency failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Fence (GPU synchronization)
    // ══════════════════════════════════════════════════════════════════════

    public static MemorySegment createFence(MemorySegment device, long initialValue, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12Fence_BYTES);
            MemorySegment pp = arena.allocate(ValueLayout.ADDRESS);
            MethodHandle mh = DxgiBindings.vtableMethod(device, DEV_CREATE_FENCE,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            int hr = (int) mh.invokeExact(device, initialValue, D3D12_FENCE_FLAG_NONE, riid, pp);
            HResult.check(hr, "ID3D12Device::CreateFence");
            return pp.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateFence failed", t);
        }
    }

    /**
     * ID3D12Fence::GetCompletedValue (vtable slot 8).
     */
    public static long fenceGetCompletedValue(MemorySegment fence) {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(fence, 8,
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            return (long) mh.invokeExact(fence);
        } catch (Throwable t) {
            throw new RuntimeException("Fence::GetCompletedValue failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Command queue operations
    // ══════════════════════════════════════════════════════════════════════

    /**
     * ID3D12CommandQueue::ExecuteCommandLists (vtable slot 10).
     */
    public static void executeCommandLists(MemorySegment queue, MemorySegment cmdList, Arena arena) {
        try {
            MemorySegment cmdLists = arena.allocate(ValueLayout.ADDRESS);
            cmdLists.set(ValueLayout.ADDRESS, 0, cmdList);
            MethodHandle mh = DxgiBindings.vtableMethod(queue, 10,
                    FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            mh.invokeExact(queue, 1, cmdLists);
        } catch (Throwable t) {
            log.error("ExecuteCommandLists failed", t);
        }
    }

    /**
     * ID3D12CommandQueue::Signal (vtable slot 14).
     */
    public static void queueSignal(MemorySegment queue, MemorySegment fence, long value)
            throws WindowsNativeException {
        try {
            MethodHandle mh = DxgiBindings.vtableMethod(queue, 14,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            int hr = (int) mh.invokeExact(queue, fence, value);
            HResult.check(hr, "ID3D12CommandQueue::Signal");
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("Queue::Signal failed", t);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // GPU sync helper
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Execute a command list and wait for the GPU to complete (blocking poll).
     * <p>
     * Creates a temporary fence, signals it from the command queue, and
     * busy-waits until the fence value reaches the signal value. The fence
     * is released regardless of success or timeout.
     *
     * @throws WindowsNativeException if closing/executing fails or the fence times out
     */
    public static void executeAndWait(MemorySegment device, MemorySegment queue,
                                      MemorySegment cmdList, Arena arena)
            throws WindowsNativeException {
        closeCommandList(cmdList);
        executeCommandLists(queue, cmdList, arena);

        MemorySegment fence = createFence(device, 0, arena);
        try {
            queueSignal(queue, fence, 1);

            // Busy-wait with timeout (MNIST is tiny, should complete in < 100ms)
            long timeoutMs = 10_000;
            long deadline = System.currentTimeMillis() + timeoutMs;
            while (fenceGetCompletedValue(fence) < 1) {
                if (System.currentTimeMillis() > deadline) {
                    throw new WindowsNativeException(
                            String.format("GPU fence timeout after %d ms – the GPU may be hung or " +
                                    "the command list may contain invalid operations", timeoutMs));
                }
                Thread.onSpinWait();
            }
            com.aresstack.windirectml.runtime.DirectMlGpuBatch.recordStandaloneFenceWait();
            WarpSubmissionStats.recordSubmitAndFenceWait();
        } finally {
            DxgiBindings.release(fence);
        }
    }

    /**
     * Submit {@code cmdList} for execution. When a
     * {@link com.aresstack.windirectml.runtime.DirectMlGpuBatch} is
     * active on the calling thread the submission is fire-and-forget –
     * the command list and {@code cmdAllocator} are {@code AddRef}'d so
     * they outlive the kernel's local lifecycle, a global UAV barrier
     * is recorded onto the list to enforce memory visibility to the next
     * submission, and the actual fence wait is deferred to
     * {@link com.aresstack.windirectml.runtime.DirectMlGpuBatch#close()}.
     * Without an active batch this behaves exactly like
     * {@link #executeAndWait(MemorySegment, MemorySegment, MemorySegment, Arena)}.
     */
    public static void executeOrDefer(MemorySegment device, MemorySegment queue,
                                      MemorySegment cmdList, MemorySegment cmdAllocator,
                                      Arena arena) throws WindowsNativeException {
        com.aresstack.windirectml.runtime.DirectMlGpuBatch batch =
                com.aresstack.windirectml.runtime.DirectMlGpuBatch.current();
        if (batch == null) {
            executeAndWait(device, queue, cmdList, arena);
            return;
        }
        // Flush UAV writes so the next batched submission observes them.
        uavBarrier(cmdList, arena);
        closeCommandList(cmdList);
        executeCommandLists(queue, cmdList, arena);
        batch.retain(cmdList, cmdAllocator);
        // GEMMA-WARP-13b-3b: one submit, no fence — the fence is coalesced into the batch drain.
        WarpSubmissionStats.recordSubmit();
    }

    /**
     * Upload float data from CPU to a GPU default-heap buffer via an upload buffer.
     * <p>
     * Backwards-compatible legacy entry point used by the Phi-3 GPU pipeline.
     * The destination is assumed to be in {@code COMMON} state on entry; it
     * decays back to {@code COMMON} after the copy (no explicit transition
     * to {@code UNORDERED_ACCESS}). New code should use
     * {@link #uploadFloatsExplicit(MemorySegment, MemorySegment, MemorySegment, float[], int, int, Arena)}
     * which makes the resource-state transitions explicit.
     */
    public static void uploadFloats(MemorySegment device, MemorySegment queue,
                                    MemorySegment dstResource, float[] data,
                                    Arena arena) throws WindowsNativeException {
        uploadFloatsInternal(device, queue, dstResource, data,
                /*stateBefore*/ -1, /*stateAfter*/ -1, arena);
    }

    /**
     * Upload raw byte data to a GPU default-heap buffer.
     * <p>
     * Used to upload packed INT4 weight data (and other non-float tensors) to GPU.
     * After this call the destination resource is in COMMON state (buffer decay rule).
     *
     * @param device      D3D12 device
     * @param queue       command queue
     * @param dstResource destination GPU buffer (default heap, COMMON state)
     * @param data        source byte array
     * @param arena       lifetime arena for temporary objects
     */
    public static void uploadBytes(MemorySegment device, MemorySegment queue,
                                   MemorySegment dstResource, byte[] data,
                                   Arena arena) throws WindowsNativeException {
        long sizeBytes = data.length;
        MemorySegment uploadBuf = createUploadBuffer(device, sizeBytes, arena);
        MemorySegment allocator = null;
        MemorySegment cmdList = null;
        try {
            MemorySegment mapped = mapResource(uploadBuf, arena);
            MemorySegment.copy(data, 0, mapped, ValueLayout.JAVA_BYTE, 0, data.length);
            unmapResource(uploadBuf);

            allocator = createCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
            cmdList = createCommandList(device, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
            copyBufferRegion(cmdList, dstResource, 0, uploadBuf, 0, sizeBytes);
            executeAndWait(device, queue, cmdList, arena);
        } finally {
            if (cmdList != null) DxgiBindings.release(cmdList);
            if (allocator != null) DxgiBindings.release(allocator);
            DxgiBindings.release(uploadBuf);
        }
    }

    /**
     * Upload raw bytes from a {@link ByteBuffer} to a GPU default-heap buffer, without first
     * materialising a host {@code byte[]}/{@code float[]}.
     * <p>
     * Heap-light upload seam (slice H2a): the {@code [position, limit)} region of {@code data} is copied
     * <b>verbatim</b> into the upload heap and then into {@code dstResource}. This accepts direct buffers
     * (including read-only {@code MappedByteBuffer} slices from {@code .wdmlpack}) and heap buffers alike;
     * the buffer's position/limit are not modified.
     * <p>
     * <b>Endianness:</b> bytes are copied byte-for-byte, so the payload must already be in the on-device
     * byte order. For the FP32 weight path that means little-endian; {@code data.order()} must be
     * {@link ByteOrder#LITTLE_ENDIAN} (enforced) to make that contract explicit and prevent accidental
     * big-endian sources. After this call {@code dstResource} is in COMMON state (buffer decay rule).
     *
     * @param device      D3D12 device
     * @param queue       command queue
     * @param dstResource destination GPU buffer (default heap, COMMON state)
     * @param data        source buffer; {@code [position, limit)} is uploaded, order must be LITTLE_ENDIAN
     * @param arena       lifetime arena for temporary objects
     */
    public static void uploadBytes(MemorySegment device, MemorySegment queue,
                                   MemorySegment dstResource, ByteBuffer data,
                                   Arena arena) throws WindowsNativeException {
        uploadBytes(device, queue, dstResource, 0L, data, arena);
    }

    /**
     * Region variant of {@link #uploadBytes(MemorySegment, MemorySegment, MemorySegment, ByteBuffer, Arena)}: copies the
     * buffer's {@code [position, limit)} verbatim into {@code dstResource} starting at byte offset
     * {@code dstByteOffset}. Used to assemble a vertically-stacked fused weight matrix from several FP32 slices without
     * a host {@code float[]} concatenation (slice item 3). LITTLE_ENDIAN is enforced; position/limit are not modified.
     */
    public static void uploadBytes(MemorySegment device, MemorySegment queue,
                                   MemorySegment dstResource, long dstByteOffset, ByteBuffer data,
                                   Arena arena) throws WindowsNativeException {
        if (data == null) {
            throw new IllegalArgumentException("data ByteBuffer must not be null");
        }
        if (data.order() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalArgumentException(
                    "ByteBuffer upload requires LITTLE_ENDIAN order, got " + data.order());
        }
        if (dstByteOffset < 0L) {
            throw new IllegalArgumentException("dstByteOffset must be >= 0: " + dstByteOffset);
        }
        long sizeBytes = data.remaining();
        if (sizeBytes <= 0L) {
            throw new IllegalArgumentException("ByteBuffer has no remaining bytes to upload");
        }
        MemorySegment uploadBuf = createUploadBuffer(device, sizeBytes, arena);
        MemorySegment allocator = null;
        MemorySegment cmdList = null;
        try {
            MemorySegment mapped = mapResource(uploadBuf, arena);
            // ofBuffer covers exactly [position, limit); read-only direct (mmap) and heap buffers are both fine.
            MemorySegment src = MemorySegment.ofBuffer(data);
            MemorySegment.copy(src, 0L, mapped, 0L, sizeBytes);
            unmapResource(uploadBuf);

            allocator = createCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
            cmdList = createCommandList(device, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
            copyBufferRegion(cmdList, dstResource, dstByteOffset, uploadBuf, 0, sizeBytes);
            executeAndWait(device, queue, cmdList, arena);
        } finally {
            if (cmdList != null) DxgiBindings.release(cmdList);
            if (allocator != null) DxgiBindings.release(allocator);
            DxgiBindings.release(uploadBuf);
        }
    }

    /**
     * Upload float data to {@code dstResource} with explicit resource-state
     * transitions around the copy.
     * <p>
     * Lifecycle recorded on the command list:
     * <pre>
     *   stateBefore → COPY_DEST     (only if stateBefore != COPY_DEST)
     *   CopyBufferRegion(upload → dst)
     *   COPY_DEST → stateAfter      (only if stateAfter != COPY_DEST)
     * </pre>
     * Pass {@code -1} for {@code stateBefore} / {@code stateAfter} to skip the
     * respective transition (legacy / implicit-promotion path).
     */
    public static void uploadFloatsExplicit(MemorySegment device, MemorySegment queue,
                                            MemorySegment dstResource, float[] data,
                                            int stateBefore, int stateAfter,
                                            Arena arena) throws WindowsNativeException {
        uploadFloatsInternal(device, queue, dstResource, data, stateBefore, stateAfter, arena);
    }

    private static void uploadFloatsInternal(MemorySegment device, MemorySegment queue,
                                             MemorySegment dstResource, float[] data,
                                             int stateBefore, int stateAfter,
                                             Arena arena) throws WindowsNativeException {
        long sizeBytes = (long) data.length * Float.BYTES;
        MemorySegment uploadBuf = createUploadBuffer(device, sizeBytes, arena);
        MemorySegment allocator = null;
        MemorySegment cmdList = null;

        try {
            // Map → copy → unmap
            MemorySegment mapped = mapResource(uploadBuf, arena);
            MemorySegment.copy(data, 0, mapped, ValueLayout.JAVA_FLOAT, 0, data.length);
            unmapResource(uploadBuf);

            // Record copy command (with optional explicit barriers)
            allocator = createCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
            cmdList = createCommandList(device, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);
            if (stateBefore >= 0 && stateBefore != D3D12_RESOURCE_STATE_COPY_DEST) {
                transitionBarrier(cmdList, dstResource, stateBefore, D3D12_RESOURCE_STATE_COPY_DEST, arena);
            }
            copyBufferRegion(cmdList, dstResource, 0, uploadBuf, 0, sizeBytes);
            if (stateAfter >= 0 && stateAfter != D3D12_RESOURCE_STATE_COPY_DEST) {
                transitionBarrier(cmdList, dstResource, D3D12_RESOURCE_STATE_COPY_DEST, stateAfter, arena);
            }
            executeAndWait(device, queue, cmdList, arena);
        } finally {
            if (cmdList != null) DxgiBindings.release(cmdList);
            if (allocator != null) DxgiBindings.release(allocator);
            DxgiBindings.release(uploadBuf);
        }
    }

    /**
     * Read float data from a GPU default-heap buffer to CPU via a readback buffer.
     * <p>
     * Backwards-compatible legacy entry point. Assumes the source is in
     * {@code UNORDERED_ACCESS} on entry and leaves it in {@code UNORDERED_ACCESS}.
     * New code should use {@link #readbackFloatsExplicit}.
     */
    public static float[] readbackFloats(MemorySegment device, MemorySegment queue,
                                         MemorySegment srcResource, int numFloats,
                                         Arena arena) throws WindowsNativeException {
        return readbackFloatsInternal(device, queue, srcResource, numFloats,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, arena);
    }

    /**
     * Read float data from {@code srcResource} with explicit resource-state
     * transitions around the copy.
     * <p>
     * Lifecycle recorded on the command list:
     * <pre>
     *   stateBefore → COPY_SOURCE     (only if stateBefore != COPY_SOURCE)
     *   CopyBufferRegion(src → readback)
     *   COPY_SOURCE → stateAfter      (only if stateAfter != COPY_SOURCE)
     * </pre>
     */
    public static float[] readbackFloatsExplicit(MemorySegment device, MemorySegment queue,
                                                 MemorySegment srcResource, int numFloats,
                                                 int stateBefore, int stateAfter,
                                                 Arena arena) throws WindowsNativeException {
        return readbackFloatsInternal(device, queue, srcResource, numFloats,
                stateBefore, stateAfter, arena);
    }

    private static float[] readbackFloatsInternal(MemorySegment device, MemorySegment queue,
                                                  MemorySegment srcResource, int numFloats,
                                                  int stateBefore, int stateAfter,
                                                  Arena arena) throws WindowsNativeException {
        long sizeBytes = (long) numFloats * Float.BYTES;
        MemorySegment readbackBuf = createReadbackBuffer(device, sizeBytes, arena);
        MemorySegment allocator = null;
        MemorySegment cmdList = null;

        try {
            allocator = createCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
            cmdList = createCommandList(device, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, arena);

            if (stateBefore >= 0 && stateBefore != D3D12_RESOURCE_STATE_COPY_SOURCE) {
                transitionBarrier(cmdList, srcResource,
                        stateBefore, D3D12_RESOURCE_STATE_COPY_SOURCE, arena);
            }
            copyBufferRegion(cmdList, readbackBuf, 0, srcResource, 0, sizeBytes);
            if (stateAfter >= 0 && stateAfter != D3D12_RESOURCE_STATE_COPY_SOURCE) {
                transitionBarrier(cmdList, srcResource,
                        D3D12_RESOURCE_STATE_COPY_SOURCE, stateAfter, arena);
            }
            executeAndWait(device, queue, cmdList, arena);

            MemorySegment mapped = mapResource(readbackBuf, arena);
            float[] result = new float[numFloats];
            MemorySegment.copy(mapped, ValueLayout.JAVA_FLOAT, 0, result, 0, numFloats);
            unmapResource(readbackBuf);
            WarpSubmissionStats.recordReadback(); // the submit was already counted by executeAndWait above
            return result;
        } finally {
            if (cmdList != null) DxgiBindings.release(cmdList);
            if (allocator != null) DxgiBindings.release(allocator);
            DxgiBindings.release(readbackBuf);
        }
    }
}
