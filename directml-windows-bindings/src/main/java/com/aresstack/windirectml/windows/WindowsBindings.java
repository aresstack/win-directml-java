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
    private String selectedBackend = "";

    /** Which DXGI adapter to bind (GEMMA-AUTO-GPU-1), independent of the {@code backend} string. */
    public enum AdapterMode { DEFAULT, WARP, HARDWARE }

    private AdapterMode adapterMode = AdapterMode.DEFAULT;
    private DxgiBindings.AdapterDesc adapterDesc; // identity of the bound adapter (null if unreadable)

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
        String b = backend == null ? "" : backend.trim();
        init(backend, "warp".equalsIgnoreCase(b) ? AdapterMode.WARP : AdapterMode.DEFAULT);
    }

    /**
     * Initialise the stack with an explicit {@link AdapterMode} (GEMMA-AUTO-GPU-1), decoupling the physical
     * DXGI adapter from the {@code backend} string. {@code backend} still drives DML/matvec behaviour
     * (e.g. {@link #isWarpBackend()}); {@code adapterMode} picks the adapter:
     * <ul>
     *   <li>{@link AdapterMode#WARP} — the explicit WARP software rasterizer (EnumWarpAdapter);</li>
     *   <li>{@link AdapterMode#HARDWARE} — the first non-software hardware adapter, or a clear error if none;</li>
     *   <li>{@link AdapterMode#DEFAULT} — DXGI adapter {@code -Dwindirectml.dxgi.adapterIndex} (default 0).</li>
     * </ul>
     * So callers can run the same {@code directml} (DML-GEMM) code path explicitly on WARP or on a hardware
     * GPU.
     */
    public void init(String backend, AdapterMode requestedAdapterMode) throws WindowsNativeException {
        if (closed) throw new IllegalStateException("WindowsBindings already closed");
        if (initialised) return;
        this.adapterMode = requestedAdapterMode == null ? AdapterMode.DEFAULT : requestedAdapterMode;
        log.info("WindowsBindings.init(backend={}, adapterMode={})", backend, this.adapterMode);
        selectedBackend = backend == null ? "" : backend.trim().toLowerCase(java.util.Locale.ROOT);

        if (!isSupported()) {
            throw new WindowsNativeException("Not running on Windows – native bindings unavailable");
        }

        // 0. Optional: enable D3D12 + DML validation layers *before* any device exists.
        if (debugRequested) {
            boolean ok = D3D12Bindings.enableDebugLayer(arena);
            log.info("DirectML debug mode requested (-Dwindirectml.debug=true); D3D12 layer enabled={}", ok);
        }

        // 1. DXGI Factory + Adapter (per the requested AdapterMode)
        int adapterIndex = Integer.getInteger("windirectml.dxgi.adapterIndex", 0);
        switch (this.adapterMode) {
            case WARP -> {
                dxgiFactory = DxgiBindings.createFactory4(arena);
                dxgiAdapter = DxgiBindings.enumWarpAdapter(dxgiFactory, arena);
            }
            case HARDWARE -> {
                dxgiFactory = DxgiBindings.createFactory1(arena);
                dxgiAdapter = selectHardwareAdapter(adapterIndex);
            }
            default -> {
                dxgiFactory = DxgiBindings.createFactory1(arena);
                dxgiAdapter = DxgiBindings.enumAdapters1(dxgiFactory, adapterIndex, arena);
                if (dxgiAdapter == null) {
                    throw new WindowsNativeException("No DXGI adapter found at index " + adapterIndex);
                }
            }
        }
        try {
            adapterDesc = DxgiBindings.getAdapterDesc1(dxgiAdapter, arena);
        } catch (WindowsNativeException de) {
            log.warn("Could not read adapter description: {}", de.getMessage());
        }
        log.info("adapter selection: mode={} adapter={} (software/warp={})",
                this.adapterMode, adapterDesc != null ? adapterDesc : "(desc unavailable)",
                adapterDesc != null && adapterDesc.software());

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

    public boolean isWarpBackend() {
        return "warp".equals(selectedBackend);
    }

    /** The adapter-selection mode this device was initialised with (GEMMA-AUTO-GPU-1). */
    public AdapterMode adapterMode() {
        return adapterMode;
    }

    /** The bound adapter's description/vendor/device/software identity, or {@code null} if unreadable. */
    public DxgiBindings.AdapterDesc adapterDesc() {
        return adapterDesc;
    }

    /** Whether the bound adapter is the software (WARP) rasterizer (GEMMA-AUTO-GPU-1). */
    public boolean isSoftwareAdapter() {
        return adapterMode == AdapterMode.WARP || (adapterDesc != null && adapterDesc.software());
    }

    /**
     * Select the first non-software (hardware) DXGI adapter from {@code startIndex} (GEMMA-AUTO-GPU-1).
     * Throws a clear, actionable error when only the software/WARP adapter is present.
     */
    private MemorySegment selectHardwareAdapter(int startIndex) throws WindowsNativeException {
        for (int i = startIndex; ; i++) {
            MemorySegment candidate = DxgiBindings.enumAdapters1(dxgiFactory, i, arena);
            if (candidate == null) {
                break;
            }
            DxgiBindings.AdapterDesc d;
            try {
                d = DxgiBindings.getAdapterDesc1(candidate, arena);
            } catch (WindowsNativeException e) {
                continue; // unreadable adapter — skip
            }
            if (!d.software()) {
                log.info("HARDWARE adapter selected at index {}: {}", i, d);
                return candidate;
            }
            log.info("Skipping software/WARP adapter at index {}: {}", i, d);
        }
        throw new WindowsNativeException("No hardware DirectML adapter found; use Backend=WARP or "
                + "configure -Dwindirectml.dxgi.adapterIndex.");
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
