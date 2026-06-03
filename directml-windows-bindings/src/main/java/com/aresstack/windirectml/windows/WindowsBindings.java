package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * High-level façade for the Windows native stack (DXGI → D3D12 → DirectML).
 * <p>
 * All FFM / Panama details are confined to this module.
 * The rest of the project (inference, runtime, encoder, sidecar) never sees
 * {@link MemorySegment}, {@link java.lang.invoke.MethodHandle},
 * or {@link java.lang.foreign.FunctionDescriptor}.
 * <p>
 * Bindings are hand-written against the Windows SDK headers using
 * Java 21 Foreign Function &amp; Memory API.
 * A {@code jextract} Gradle task can regenerate machine-generated
 * equivalents from {@code DirectML.h}, {@code d3d12.h}, {@code dxgi.h}.
 * <p>
 * The JVM <b>must</b> be started with {@code --enable-native-access=ALL-UNNAMED}.
 *
 * @see DxgiBindings
 * @see D3D12Bindings
 * @see DirectMlBindings
 * @see <a href="https://learn.microsoft.com/en-us/windows/ai/directml/">DirectML</a>
 */
public final class WindowsBindings implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(WindowsBindings.class);

    private final Arena arena;

    // COM pointers – set during init()
    private MemorySegment dxgiFactory;
    private MemorySegment dxgiAdapter;
    private MemorySegment d3d12Device;
    private MemorySegment commandQueue;
    private MemorySegment dmlDevice;
    /**
     * Raw {@code DML_FEATURE_LEVEL} of {@link #dmlDevice}; {@code 0} when no DML device.
     */
    private int dmlFeatureLevel;
    /**
     * Non-null when the debug layer is active and the device exposes an info queue.
     */
    private MemorySegment infoQueue;
    /**
     * True when {@code -Dwindirectml.debug=true} (or constructor override) requested the debug layer.
     */
    private final boolean debugRequested;

    private boolean initialised = false;
    private boolean closed = false;

    public WindowsBindings() {
        this(Boolean.getBoolean("windirectml.debug"));
    }

    public WindowsBindings(boolean debug) {
        this.arena = Arena.ofShared();
        this.debugRequested = debug;
    }

    /**
     * Check whether the current platform supports the Windows native path.
     */
    public static boolean isSupported() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("windows");
    }

    /**
     * Initialise the full DXGI → D3D12 → DirectML stack.
     * <ol>
     *   <li>Create a DXGI factory and pick the first hardware adapter</li>
     *   <li>For "warp" backend, use IDXGIFactory4::EnumWarpAdapter</li>
     *   <li>Create a D3D12 device on that adapter</li>
     *   <li>Create a D3D12 DIRECT command queue</li>
     *   <li>Create a DirectML device on top of D3D12</li>
     * </ol>
     *
     * @param backend "directml", "auto", "warp", or "cpu"
     * @throws WindowsNativeException if initialisation fails
     * @throws IllegalStateException  if already closed
     */
    public void init(String backend) throws WindowsNativeException {
        if (closed) throw new IllegalStateException("WindowsBindings already closed");
        if (initialised) return;
        log.info("WindowsBindings.init(backend={})", backend);

        if (!isSupported()) {
            throw new WindowsNativeException("Not running on Windows – native bindings unavailable");
        }

        // 0. Optional: enable D3D12 + DML validation layers *before* any device exists.
        if (debugRequested) {
            boolean ok = D3D12Bindings.enableDebugLayer(arena);
            log.info("DirectML debug mode requested (-Dwindirectml.debug=true); D3D12 layer enabled={}", ok);
        }

        // 1. DXGI Factory + Adapter
        int adapterIndex = Integer.getInteger("windirectml.dxgi.adapterIndex", 0);

        if ("warp".equalsIgnoreCase(backend)) {
            // WARP path: Factory4 → EnumWarpAdapter
            dxgiFactory = DxgiBindings.createFactory4(arena);
            dxgiAdapter = DxgiBindings.enumWarpAdapter(dxgiFactory, arena);
            log.info("Using WARP (software) adapter");
        } else {
            // Standard path: Factory1 → EnumAdapters1
            dxgiFactory = DxgiBindings.createFactory1(arena);
            dxgiAdapter = DxgiBindings.enumAdapters1(dxgiFactory, adapterIndex, arena);
            if (dxgiAdapter == null) {
                throw new WindowsNativeException("No DXGI adapter found at index " + adapterIndex);
            }
            log.info("Using DXGI adapter at index {}: {}", adapterIndex, dxgiAdapter);
        }

        // 2. D3D12 device
        d3d12Device = D3D12Bindings.createDevice(
                dxgiAdapter, D3D12Bindings.D3D_FEATURE_LEVEL_11_0, arena);

        // 2b. With debug layer active, grab the info queue
        if (debugRequested) {
            infoQueue = D3D12Bindings.queryInfoQueue(d3d12Device, arena);
            log.info("D3D12 info queue available: {}", infoQueue != null);
        }

        // 3. Command queue
        commandQueue = D3D12Bindings.createCommandQueue(d3d12Device, arena);

        // 4. DirectML device (even for WARP – WARP runs DML on CPU)
        if ("cpu".equalsIgnoreCase(backend)) {
            log.info("CPU-only mode requested – skipping DirectML device creation");
        } else {
            int dmlFlags = debugRequested
                    ? DirectMlBindings.DML_CREATE_DEVICE_FLAG_DEBUG
                    : DirectMlBindings.DML_CREATE_DEVICE_FLAG_NONE;
            try {
                dmlDevice = DirectMlBindings.createDevice(d3d12Device, dmlFlags, arena);
                try {
                    dmlFeatureLevel = DirectMlBindings.queryMaxFeatureLevel(dmlDevice, arena);
                    log.info("DirectML.dll source: {}, max DML_FEATURE_LEVEL = {} (raw 0x{})",
                            DirectMlBindings.directMlSource(),
                            DirectMlBindings.formatFeatureLevel(dmlFeatureLevel),
                            Integer.toHexString(dmlFeatureLevel));
                } catch (WindowsNativeException fe) {
                    log.warn("Could not query DML feature level: {}", fe.getMessage());
                }
            } catch (WindowsNativeException e) {
                if ("auto".equalsIgnoreCase(backend)) {
                    log.warn("DirectML device creation failed (auto mode), continuing without: {}",
                            e.getMessage());
                } else {
                    throw e;
                }
            }
        }

        initialised = true;
        log.info("WindowsBindings initialised: backend={}, d3d12={}, dml={}",
                backend, d3d12Device, dmlDevice != null ? dmlDevice : "(none)");
    }

    // ── Accessors for downstream code ────────────────────────────────────

    public MemorySegment getDxgiFactory() {
        return dxgiFactory;
    }

    public MemorySegment getDxgiAdapter() {
        return dxgiAdapter;
    }

    public MemorySegment getD3d12Device() {
        return d3d12Device;
    }

    public MemorySegment getCommandQueue() {
        return commandQueue;
    }

    public MemorySegment getDmlDevice() {
        return dmlDevice;
    }

    public boolean isInitialised() {
        return initialised;
    }

    public boolean hasDirectMl() {
        return dmlDevice != null;
    }

    /**
     * @return raw {@code DML_FEATURE_LEVEL} value (e.g. {@code 0x6100}), or {@code 0} if no DML device.
     */
    public int getDmlFeatureLevel() {
        return dmlFeatureLevel;
    }

    /**
     * Drain pending D3D12/DirectML debug messages from the device's info queue.
     * <p>
     * Returns an empty list when the debug layer is not active or the info
     * queue is unavailable. Useful to enrich an exception message right after
     * a failing native call:
     * <pre>
     *   try {
     *       DirectMlBindings.createOperator(...);
     *   } catch (WindowsNativeException e) {
     *       for (String msg : wb.drainDebugMessages()) log.error(msg);
     *       throw e;
     *   }
     * </pre>
     */
    public java.util.List<String> drainDebugMessages() {
        if (infoQueue == null) return java.util.List.of();
        return D3D12Bindings.drainInfoQueue(infoQueue, arena);
    }

    public boolean isDebugEnabled() {
        return debugRequested && infoQueue != null;
    }

    @Override
    public void close() {
        if (closed) return;           // idempotent
        closed = true;
        if (!initialised) return;
        log.info("WindowsBindings closing – releasing COM objects");

        // Release in reverse creation order; null-safe
        safeRelease(dmlDevice, "DML device");
        safeRelease(commandQueue, "command queue");
        safeRelease(infoQueue, "info queue");
        safeRelease(d3d12Device, "D3D12 device");
        safeRelease(dxgiAdapter, "DXGI adapter");
        safeRelease(dxgiFactory, "DXGI factory");

        dmlDevice = null;
        commandQueue = null;
        infoQueue = null;
        d3d12Device = null;
        dxgiAdapter = null;
        dxgiFactory = null;

        arena.close();
        initialised = false;
        log.info("WindowsBindings closed");
    }

    /**
     * Null-safe COM Release with error logging (never throws).
     */
    private static void safeRelease(MemorySegment comPtr, String label) {
        if (comPtr == null || comPtr.equals(MemorySegment.NULL)) return;
        try {
            DxgiBindings.release(comPtr);
        } catch (Exception e) {
            log.warn("Failed to release {} COM object: {}", label, e.getMessage());
        }
    }
}
