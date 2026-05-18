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
 * Upload/Download laufen über die bestehenden, von der Phi-3- und
 * MNIST-Pipeline genutzten Helfer
 * {@link D3D12Bindings#uploadFloats(MemorySegment, MemorySegment, MemorySegment, float[], Arena)}
 * bzw.
 * {@link D3D12Bindings#readbackFloats(MemorySegment, MemorySegment, MemorySegment, int, Arena)}.
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
            D3D12Bindings.uploadFloats(device, queue, resource, floats, scratch);
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
            float[] out = D3D12Bindings.readbackFloats(device, queue, resource, n, scratch);
            ByteBuffer data = destination.data();
            data.order(ByteOrder.LITTLE_ENDIAN);
            FloatBuffer view = data.duplicate().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            view.position(0);
            view.put(out, 0, n);
        } catch (WindowsNativeException e) {
            throw new RuntimeException("DefaultGpuBuffer.download failed", e);
        }
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

