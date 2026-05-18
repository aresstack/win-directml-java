package com.aresstack.windirectml.runtime;

import com.aresstack.windirectml.windows.D3D12Bindings;
import com.aresstack.windirectml.windows.DxgiBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Konkrete {@link GpuBuffer}-Implementierung auf Basis einer
 * D3D12-Default-Heap-Resource ({@code D3D12_HEAP_TYPE_DEFAULT}) mit
 * {@code UNORDERED_ACCESS}-Flag, sodass DirectML-Operatoren den Buffer
 * direkt als Tensor-Backing verwenden können.
 * <p>
 * <b>Resource-State-Invariante:</b> Direkt nach jeder Public-Operation
 * ({@link #upload(CpuTensor)}, {@link #download(CpuTensor)}) ist die
 * D3D12-Resource im Zustand {@code D3D12_RESOURCE_STATE_UNORDERED_ACCESS} –
 * dem Steady-State, in dem DirectML-Kernel binden. Frisch allokierte Buffer
 * starten im Zustand {@code COMMON}; der erste {@code upload()} transitioniert
 * von dort nach {@code UAV}.
 * <p>
 * Upload/Download laufen über die expliziten Helfer
 * {@link D3D12Bindings#uploadFloatsExplicit} bzw.
 * {@link D3D12Bindings#readbackFloatsExplicit}, damit der Debug Layer beim
 * ersten echten Kernel-Dispatch nicht über implizite Promotion stolpert.
 * <p>
 * Aktuell wird nur {@link TensorDataType#FLOAT32} produktiv unterstützt;
 * weitere Datentypen folgen, sobald die Kernel sie benötigen.
 */
public final class DefaultGpuBuffer implements GpuBuffer {

    private static final Logger log = LoggerFactory.getLogger(DefaultGpuBuffer.class);

    private final MemorySegment device;
    private final MemorySegment queue;
    private final MemorySegment resource;
    private final long sizeInBytes;
    private final BufferUsage usage;

    /** Aktueller D3D12-Resource-State. Wird von upload/download fortgeschrieben. */
    private int currentState;
    private boolean closed = false;

    DefaultGpuBuffer(MemorySegment device,
                     MemorySegment queue,
                     MemorySegment resource,
                     long sizeInBytes,
                     BufferUsage usage) {
        this.device = device;
        this.queue = queue;
        this.resource = resource;
        this.sizeInBytes = sizeInBytes;
        this.usage = usage;
        this.currentState = D3D12Bindings.D3D12_RESOURCE_STATE_COMMON;
    }

    /**
     * Roher D3D12-Resource-Pointer. Wird von DirectML-Kernel-Implementierungen
     * benötigt, um den Buffer in Binding-Tabellen einzuhängen.
     */
    public MemorySegment resource() {
        return resource;
    }

    /**
     * GPU-Virtual-Address dieses Buffers ({@code ID3D12Resource::GetGPUVirtualAddress}).
     */
    public long gpuVirtualAddress() {
        return D3D12Bindings.getGpuVirtualAddress(resource);
    }

    @Override
    public long sizeInBytes() {
        return sizeInBytes;
    }

    @Override
    public BufferUsage usage() {
        return usage;
    }

    @Override
    public void upload(CpuTensor source) {
        ensureOpen();
        long required = (long) source.shape().elementCount() * source.dataType().byteWidth();
        if (required > sizeInBytes) {
            throw new IllegalArgumentException(
                    "source tensor needs " + required + " bytes but buffer has " + sizeInBytes);
        }
        if (source.dataType() != TensorDataType.FLOAT32) {
            throw new UnsupportedOperationException(
                    "DefaultGpuBuffer.upload currently supports FLOAT32 only, was " + source.dataType());
        }
        float[] floats = toFloatArray(source.data(), (int) source.shape().elementCount());
        try (Arena scratch = Arena.ofConfined()) {
            // Explicit transitions: currentState → COPY_DEST → UAV. After upload we
            // park the resource in UAV, the steady-state for DirectML dispatches.
            D3D12Bindings.uploadFloatsExplicit(
                    device, queue, resource, floats,
                    currentState,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    scratch);
            currentState = D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        } catch (WindowsNativeException e) {
            throw new RuntimeException("DefaultGpuBuffer.upload failed", e);
        }
    }

    @Override
    public void download(CpuTensor destination) {
        ensureOpen();
        long required = (long) destination.shape().elementCount() * destination.dataType().byteWidth();
        if (required > sizeInBytes) {
            throw new IllegalArgumentException(
                    "destination tensor needs " + required + " bytes but buffer has " + sizeInBytes);
        }
        if (destination.dataType() != TensorDataType.FLOAT32) {
            throw new UnsupportedOperationException(
                    "DefaultGpuBuffer.download currently supports FLOAT32 only, was " + destination.dataType());
        }
        int n = (int) destination.shape().elementCount();
        try (Arena scratch = Arena.ofConfined()) {
            // Explicit transitions: currentState → COPY_SOURCE → UAV. After download
            // the buffer remains in UAV so a subsequent kernel can re-use it.
            float[] out = D3D12Bindings.readbackFloatsExplicit(
                    device, queue, resource, n,
                    currentState,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    scratch);
            currentState = D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ByteBuffer data = destination.data();
            data.order(ByteOrder.LITTLE_ENDIAN);
            FloatBuffer view = data.duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            view.position(0);
            view.put(out, 0, n);
        } catch (WindowsNativeException e) {
            throw new RuntimeException("DefaultGpuBuffer.download failed", e);
        }
    }

    /**
     * Aktuell bekannter D3D12-Resource-State dieses Buffers.
     * <p>
     * Kernel-Implementierungen, die den Buffer direkt einbinden, sollten diesen
     * Wert als Ausgangspunkt für eigene Barriers nutzen.
     */
    public int currentResourceState() {
        return currentState;
    }

    /**
     * Setze den aktuellen D3D12-Resource-State extern. Wird von Kernel-Code
     * verwendet, der den Buffer in einem eigenen Command-List-Recording
     * weiter transitioniert.
     */
    public void setCurrentResourceState(int state) {
        this.currentState = state;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            DxgiBindings.release(resource);
        } catch (RuntimeException e) {
            log.warn("DefaultGpuBuffer.close: release failed for {}: {}", resource, e.getMessage());
        }
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("DefaultGpuBuffer already closed");
    }

    private static float[] toFloatArray(ByteBuffer buf, int n) {
        ByteBuffer le = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        le.position(0);
        FloatBuffer fb = le.asFloatBuffer();
        float[] out = new float[n];
        fb.get(out, 0, n);
        return out;
    }
}

