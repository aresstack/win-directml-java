package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * High-level façade for the Windows native stack (DXGI → D3D12 → DirectML).
 * <p>
 * All FFM / Panama details are confined to this module.
 * The rest of the project (inference, graph, ACP) never sees
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

    private boolean initialised = false;
    private boolean closed = false;

    public WindowsBindings() {
        this.arena = Arena.ofShared();
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
     *   <li>Create a D3D12 device on that adapter</li>
     *   <li>Create a D3D12 DIRECT command queue</li>
     *   <li>Create a DirectML device on top of D3D12</li>
     * </ol>
     *
     * @param backend "directml", "cpu", or "auto"
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

// 1. DXGI Factory
        int adapterIndex = Integer.getInteger("windirectml.dxgi.adapterIndex",
                Integer.getInteger("winacp.dxgi.adapterIndex", 0));

        dxgiFactory = DxgiBindings.createFactory1(arena);

// 2. First hardware adapter
        dxgiAdapter = DxgiBindings.enumAdapters1(dxgiFactory, adapterIndex, arena);
        if (dxgiAdapter == null) {
            throw new WindowsNativeException("No DXGI adapter found at index " + adapterIndex);
        }
        log.info("Using DXGI adapter at index {}: {}", adapterIndex, dxgiAdapter);

        // 3. D3D12 device
        d3d12Device = D3D12Bindings.createDevice(
                dxgiAdapter, D3D12Bindings.D3D_FEATURE_LEVEL_11_0, arena);

        // 4. Command queue (needed for DirectML dispatch)
        commandQueue = D3D12Bindings.createCommandQueue(d3d12Device, arena);

        // 5. DirectML device
        if ("cpu".equalsIgnoreCase(backend)) {
            log.info("CPU-only mode requested – skipping DirectML device creation");
        } else {
            try {
                dmlDevice = DirectMlBindings.createDevice(
                        d3d12Device, DirectMlBindings.DML_CREATE_DEVICE_FLAG_NONE, arena);
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
        log.info("WindowsBindings initialised: d3d12={}, dml={}",
                d3d12Device, dmlDevice != null ? dmlDevice : "(none)");
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

    @Override
    public void close() {
        if (closed) return;           // idempotent
        closed = true;
        if (!initialised) return;
        log.info("WindowsBindings closing – releasing COM objects");

        // Release in reverse creation order; null-safe
        safeRelease(dmlDevice, "DML device");
        safeRelease(commandQueue, "command queue");
        safeRelease(d3d12Device, "D3D12 device");
        safeRelease(dxgiAdapter, "DXGI adapter");
        safeRelease(dxgiFactory, "DXGI factory");

        dmlDevice = null;
        commandQueue = null;
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
