package com.aresstack.windirectml.runtime;

/**
 * Abstrakter GPU-Buffer.
 * <p>
 * Kapselt eine D3D12-Resource. Die Runtime-Schicht arbeitet mit dieser
 * Abstraktion, damit Encoder- und Decoder-Code nicht direkt gegen die
 * FFM-Bindings programmieren müssen.
 *
 * @see DirectMlContext#allocateBuffer(long, BufferUsage)
 */
public interface GpuBuffer extends AutoCloseable {

    /**
     * Größe in Bytes.
     */
    long sizeInBytes();

    BufferUsage usage();

    /**
     * Lade Daten vom CPU-Puffer auf die GPU.
     */
    void upload(CpuTensor source);

    /**
     * Lade Daten von der GPU in einen CPU-Puffer.
     */
    void download(CpuTensor destination);

    @Override
    void close();

    enum BufferUsage {
        /**
         * Schreibgeschützte Gewichte, Upload einmalig beim Modell-Load.
         */
        WEIGHT,
        /**
         * Aktivierungen, frequenter Read/Write.
         */
        ACTIVATION,
        /**
         * Staging-Buffer für Upload/Download.
         */
        STAGING
    }
}

