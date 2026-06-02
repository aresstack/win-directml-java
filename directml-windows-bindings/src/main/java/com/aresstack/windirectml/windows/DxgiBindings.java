package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Java 21 FFM bindings for {@code dxgi.dll} – DXGI Factory and Adapter enumeration.
 * <p>
 * These bindings call directly into the Windows system DLL using
 * {@link Linker} and {@link SymbolLookup}. No third-party library involved.
 * <p>
 * When jextract-generated sources are available they supersede this class,
 * but the hand-written version serves as the bootstrap / fallback.
 *
 * <h3>Bound functions</h3>
 * <ul>
 *   <li>{@code HRESULT CreateDXGIFactory1(REFIID riid, void **ppFactory)}</li>
 *   <li>IDXGIFactory1 vtable: {@code EnumAdapters1}, {@code Release}</li>
 * </ul>
 *
 * @see <a href="https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-createdxgifactory1">CreateDXGIFactory1</a>
 */
public final class DxgiBindings {

    private static final Logger log = LoggerFactory.getLogger(DxgiBindings.class);

    private DxgiBindings() {
    }

    // ── Function descriptors ─────────────────────────────────────────────

    /**
     * {@code HRESULT CreateDXGIFactory1(REFIID riid, void **ppFactory)}
     */
    private static final FunctionDescriptor CREATE_DXGI_FACTORY1_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,           // HRESULT return
                    ValueLayout.ADDRESS,             // REFIID  riid
                    ValueLayout.ADDRESS              // void ** ppFactory
            );

    // ── Vtable slot indices (IDXGIFactory1 inherits IDXGIFactory) ────────
    // IUnknown: QueryInterface(0), AddRef(1), Release(2)
    // IDXGIObject: SetPrivateData(3), SetPrivateDataInterface(4), GetPrivateData(5), GetParent(6)
    // IDXGIFactory: EnumAdapters(7), MakeWindowAssociation(8), GetWindowAssociation(9), CreateSwapChain(10), CreateSoftwareAdapter(11)
    // IDXGIFactory1: EnumAdapters1(12), IsCurrent(13)
    static final int VTABLE_ADDREF = 1;
    static final int VTABLE_RELEASE = 2;
    static final int VTABLE_ENUM_ADAPTERS1 = 12;

    // ── Lazy-initialised handles ─────────────────────────────────────────

    private static volatile MethodHandle createDxgiFactory1Handle;

    /**
     * Look up {@code CreateDXGIFactory1} in {@code dxgi.dll}.
     */
    private static MethodHandle getCreateDxgiFactory1Handle() {
        if (createDxgiFactory1Handle == null) {
            synchronized (DxgiBindings.class) {
                if (createDxgiFactory1Handle == null) {
                    SymbolLookup dxgi = SymbolLookup.libraryLookup("dxgi.dll", Arena.global());
                    MemorySegment addr = dxgi.find("CreateDXGIFactory1")
                            .orElseThrow(() -> new UnsatisfiedLinkError("CreateDXGIFactory1 not found in dxgi.dll"));
                    createDxgiFactory1Handle = Linker.nativeLinker()
                            .downcallHandle(addr, CREATE_DXGI_FACTORY1_DESC);
                    log.debug("Resolved CreateDXGIFactory1 at {}", addr);
                }
            }
        }
        return createDxgiFactory1Handle;
    }

    // ── Public API ───────────────────────────────────────────────────────

    /**
     * Call {@code CreateDXGIFactory1(&IID_IDXGIFactory1, &pFactory)} and
     * return the raw COM pointer as a {@link MemorySegment}.
     *
     * @param arena confined arena that owns the allocated memory
     * @return COM pointer to {@code IDXGIFactory1}
     */
    public static MemorySegment createFactory1(Arena arena) throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_IDXGIFactory1_BYTES);
            MemorySegment ppFactory = arena.allocate(ValueLayout.ADDRESS);

            int hr = (int) getCreateDxgiFactory1Handle().invokeExact(riid, ppFactory);
            HResult.check(hr, "CreateDXGIFactory1");

            MemorySegment factory = ppFactory.get(ValueLayout.ADDRESS, 0)
                    .reinterpret(Long.MAX_VALUE);
            log.info("IDXGIFactory1 created: {}", factory);
            return factory;
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateDXGIFactory1 invocation failed", t);
        }
    }

    /**
     * Enumerate GPU adapters via {@code IDXGIFactory1::EnumAdapters1}.
     *
     * @param factory COM pointer to IDXGIFactory1
     * @param index   adapter index (0-based)
     * @param arena   arena for allocations
     * @return COM pointer to IDXGIAdapter1, or {@code null} if index is out of range
     */
    public static MemorySegment enumAdapters1(MemorySegment factory, int index, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment ppAdapter = arena.allocate(ValueLayout.ADDRESS);

            // IDXGIFactory1::EnumAdapters1(this, UINT index, IDXGIAdapter1 **ppAdapter)
            MethodHandle enumAdapters1 = vtableMethod(factory, VTABLE_ENUM_ADAPTERS1,
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,   // HRESULT
                            ValueLayout.ADDRESS,     // this
                            ValueLayout.JAVA_INT,   // UINT index
                            ValueLayout.ADDRESS      // IDXGIAdapter1 **ppAdapter
                    ));

            int hr = (int) enumAdapters1.invokeExact(factory, index, ppAdapter);

            // DXGI_ERROR_NOT_FOUND (0x887A0002) means no more adapters
            if (hr == (int) 0x887A0002L) {
                return null;
            }
            HResult.check(hr, "IDXGIFactory1::EnumAdapters1(" + index + ")");

            return ppAdapter.get(ValueLayout.ADDRESS, 0).reinterpret(Long.MAX_VALUE);
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("EnumAdapters1 invocation failed", t);
        }
    }

    /**
     * Call {@code IUnknown::Release} on a COM object.
     */
    public static void release(MemorySegment comObject) {
        try {
            MethodHandle releaseHandle = vtableMethod(comObject, VTABLE_RELEASE,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            int refCount = (int) releaseHandle.invokeExact(comObject);
            log.trace("Release → refCount={}", refCount);
        } catch (Throwable t) {
            log.warn("COM Release failed", t);
        }
    }

    /**
     * Call {@code IUnknown::AddRef} on a COM object. Used by
     * {@link com.aresstack.windirectml.runtime.DirectMlGpuBatch} to keep
     * command-lists and allocators alive past the kernel's local
     * {@code try/finally} until the deferred GPU drain completes.
     */
    public static void addRef(MemorySegment comObject) {
        try {
            MethodHandle addRefHandle = vtableMethod(comObject, VTABLE_ADDREF,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            int refCount = (int) addRefHandle.invokeExact(comObject);
            log.trace("AddRef → refCount={}", refCount);
        } catch (Throwable t) {
            log.warn("COM AddRef failed", t);
        }
    }

    // ── COM vtable helper ────────────────────────────────────────────────

    /**
     * Cache for downcall MethodHandles, keyed by the native function pointer address.
     * <p>
     * <b>This is the single most important performance optimization in the project.</b>
     * Without this cache, every COM vtable call (D3D12, DirectML) creates a new
     * {@link Linker#downcallHandle} which takes 5-50 µs. In the Phi-3 decode loop,
     * that means ~2000 handle creations per token, adding 10-100 ms of pure overhead.
     * With the cache, only the first call per function pays that cost.
     */
    private static final ConcurrentHashMap<Long, MethodHandle> vtableCache = new ConcurrentHashMap<>();

    /**
     * Read a function pointer from a COM object's vtable at the given slot index.
     * <p>
     * COM objects start with a pointer to their vtable. Each vtable entry
     * is a function pointer (8 bytes on 64-bit Windows).
     * <p>
     * The resulting {@link MethodHandle} is cached by native function pointer address
     * so that repeated calls to the same vtable slot avoid the expensive
     * {@code Linker.downcallHandle()} creation.
     *
     * @param comObject pointer to the COM interface
     * @param slotIndex 0-based vtable slot index
     * @param desc      function descriptor for the target method
     * @return a downcall {@link MethodHandle} bound to the vtable slot
     */
    static MethodHandle vtableMethod(MemorySegment comObject, int slotIndex,
                                     FunctionDescriptor desc) {
        // *comObject → vtable pointer
        MemorySegment vtablePtr = comObject.get(ValueLayout.ADDRESS, 0)
                .reinterpret(Long.MAX_VALUE);
        // vtable[slotIndex] → function pointer
        MemorySegment fnPtr = vtablePtr.get(ValueLayout.ADDRESS,
                        (long) slotIndex * ValueLayout.ADDRESS.byteSize())
                .reinterpret(Long.MAX_VALUE);
        long key = fnPtr.address();
        return vtableCache.computeIfAbsent(key,
                k -> Linker.nativeLinker().downcallHandle(fnPtr, desc));
    }
}

