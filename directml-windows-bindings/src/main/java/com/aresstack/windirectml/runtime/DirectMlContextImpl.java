package com.aresstack.windirectml.runtime;

import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * Erste konkrete {@link DirectMlContext}-Implementierung.
 * <p>
 * Wrappt die bereits produktive {@link WindowsBindings}-Fassade (DXGI →
 * D3D12 → DirectML, vom Phi-3-Summarizer genutzt) und liefert darüber
 * GPU-Buffer-Allokation an Kernel- und Encoder-Code.
 * <p>
 * Verantwortlichkeiten:
 * <ul>
 *     <li>Init der Native-Stacks (delegiert an {@link WindowsBindings})</li>
 *     <li>Allokation von {@link DefaultGpuBuffer}-Instanzen über
 *         {@link D3D12Bindings#createDefaultBuffer(MemorySegment, long, Arena)}</li>
 *     <li>Buchführung über offene Buffer für sauberes Close()</li>
 * </ul>
 * <p>
 * Kernel-Dispatch (Linear, LayerNorm, GELU, Attention, …) wird in
 * eigenen Kernel-Klassen implementiert, die diese Context-Instanz
 * konsumieren – nicht hier.
 */
public final class DirectMlContextImpl implements DirectMlContext {

    private static final Logger log = LoggerFactory.getLogger(DirectMlContextImpl.class);

    private final WindowsBindings bindings;
    private final boolean ownsBindings;
    private final Arena arena;
    private final List<DefaultGpuBuffer> liveBuffers = new ArrayList<>();
    private final String backend;
    private boolean closed = false;

    /**
     * Erzeugt einen Context, der seine eigene {@link WindowsBindings} initialisiert.
     *
     * @param backend "directml", "cpu" oder "auto" (siehe {@link WindowsBindings#init(String)})
     */
    public DirectMlContextImpl(String backend) {
        this.bindings = new WindowsBindings();
        this.ownsBindings = true;
        this.arena = Arena.ofShared();
        this.backend = backend == null ? "auto" : backend;
    }

    /**
     * Erzeugt einen Context auf einer bereits initialisierten
     * {@link WindowsBindings} – nützlich, um den Phi-3-Pfad und den
     * Encoder-Pfad denselben D3D12/DML-Stack teilen zu lassen.
     */
    public DirectMlContextImpl(WindowsBindings existing) {
        this.bindings = existing;
        this.ownsBindings = false;
        this.arena = Arena.ofShared();
        this.backend = "external";
    }

    @Override
    public void initialize() throws DirectMlRuntimeException {
        if (closed) throw new DirectMlRuntimeException("Context already closed");
        if (bindings.isInitialised()) return;
        try {
            bindings.init(backend);
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException("WindowsBindings init failed (backend=" + backend + ")", e);
        }
        log.info("DirectMlContextImpl initialised: backend={}, hasDirectMl={}",
                backend, bindings.hasDirectMl());
    }

    @Override
    public boolean isReady() {
        return !closed && bindings.isInitialised();
    }

    @Override
    public String adapterDescription() {
        // DXGI adapter description ist heute nur als COM-Pointer verfügbar;
        // Bis ein dedizierter `DXGI_ADAPTER_DESC1`-Reader bindet, geben wir
        // eine stabile, eindeutige Kennzeichnung zurück.
        if (bindings.getDxgiAdapter() == null) return "(uninitialized)";
        return "DXGI adapter @ " + bindings.getDxgiAdapter()
                + (bindings.hasDirectMl() ? " (DirectML)" : " (D3D12 only)");
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, GpuBuffer.BufferUsage usage)
            throws DirectMlRuntimeException {
        ensureReady();
        if (sizeInBytes <= 0) throw new IllegalArgumentException("sizeInBytes must be > 0");
        try {
            MemorySegment resource = D3D12Bindings.createDefaultBuffer(
                    bindings.getD3d12Device(), sizeInBytes, arena);
            DefaultGpuBuffer buf = new DefaultGpuBuffer(
                    bindings.getD3d12Device(),
                    bindings.getCommandQueue(),
                    resource,
                    sizeInBytes,
                    usage);
            liveBuffers.add(buf);
            return buf;
        } catch (WindowsNativeException e) {
            throw new DirectMlRuntimeException(
                    "Failed to allocate GPU buffer (size=" + sizeInBytes + ", usage=" + usage + ")", e);
        }
    }

    @Override
    public GpuBuffer allocateBufferFor(CpuTensor tensor, GpuBuffer.BufferUsage usage)
            throws DirectMlRuntimeException {
        long bytes = (long) tensor.shape().elementCount() * tensor.dataType().byteWidth();
        if (bytes <= 0) {
            throw new IllegalArgumentException(
                    "Cannot derive buffer size from tensor dataType=" + tensor.dataType());
        }
        return allocateBuffer(bytes, usage);
    }

    @Override
    public void waitForIdle() {
        // Die heutigen Upload/Readback-Helfer warten bereits selbst via Fence
        // (D3D12Bindings.executeAndWait). Sobald Kernel-Dispatch asynchron
        // wird, ergänzt diese Methode ein explizites Fence-Signal/Wait.
    }

    /**
     * Zugriff auf die unterliegenden FFM-Bindings für Kernel-Implementierungen.
     */
    public WindowsBindings bindings() {
        return bindings;
    }

    /**
     * Geteilte Arena für lebensdauergebundene FFM-Allokationen (Descriptors,
     * Heaps, Binding-Tables). Lebt bis zum {@link #close()} dieses Contexts.
     */
    public Arena arena() {
        return arena;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        // Buffer zuerst freigeben, bevor die Devices geschlossen werden.
        for (int i = liveBuffers.size() - 1; i >= 0; i--) {
            try {
                liveBuffers.get(i).close();
            } catch (RuntimeException e) {
                log.warn("Closing live buffer failed: {}", e.getMessage());
            }
        }
        liveBuffers.clear();
        try {
            arena.close();
        } catch (RuntimeException e) {
            log.warn("Context arena close failed: {}", e.getMessage());
        }
        if (ownsBindings) {
            bindings.close();
        }
        log.info("DirectMlContextImpl closed");
    }

    private void ensureReady() throws DirectMlRuntimeException {
        if (closed) throw new DirectMlRuntimeException("Context already closed");
        if (!bindings.isInitialised()) {
            throw new DirectMlRuntimeException("Context not initialised – call initialize() first");
        }
    }
}

