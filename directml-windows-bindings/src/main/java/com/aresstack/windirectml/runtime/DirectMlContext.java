package com.aresstack.windirectml.runtime;

/**
 * High-Level-Lebenszyklus eines DirectML-Backends.
 * <p>
 * Abstrahiert über:
 * <ul>
 *   <li>D3D12 Device + Adapter</li>
 *   <li>DirectML Device</li>
 *   <li>Command Queue / Allocator / List</li>
 *   <li>Descriptor Heap</li>
 *   <li>GPU-Buffer-Verwaltung</li>
 * </ul>
 * Konkrete Implementierungen leben in {@code directml-windows-bindings}
 * und kapseln die FFM-Aufrufe gegen {@code dxgi.dll}, {@code d3d12.dll}
 * und {@code DirectML.dll}.
 * <p>
 * <b>Status:</b> Phi-3 verwendet aktuell noch die ältere
 * {@link com.aresstack.windirectml.windows.WindowsBindings}-API direkt.
 * Die Migration auf {@code DirectMlContext} erfolgt schrittweise, sobald
 * der Encoder-Pfad steht.
 */
public interface DirectMlContext extends AutoCloseable {

    /**
     * Initialisierung (Device-Auswahl, DML-Erzeugung).
     */
    void initialize() throws DirectMlRuntimeException;

    boolean isReady();

    String adapterDescription();

    /**
     * Reserviert einen GPU-Buffer der gewünschten Größe und Nutzungsklasse.
     */
    GpuBuffer allocateBuffer(long sizeInBytes, GpuBuffer.BufferUsage usage) throws DirectMlRuntimeException;

    /**
     * Convenience: Buffer passend zu einem CPU-Tensor reservieren.
     */
    GpuBuffer allocateBufferFor(CpuTensor tensor, GpuBuffer.BufferUsage usage) throws DirectMlRuntimeException;

    /**
     * Wartet auf Fertigstellung aller bisher abgeschickten GPU-Arbeit.
     */
    void waitForIdle();

    @Override
    void close();
}

