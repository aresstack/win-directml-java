package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WindowsNativeException;

/**
 * GPU-resident key/value cache for the Gemma 3 native WARP decode session (GEMMA-WARP-13c): one
 * {@link Gemma3WarpResidentKvLayerCache} per layer, so the resident decode step appends each new token's
 * k/v on the GPU and the attention kernels read them in place — the per-layer host readback/upload of the
 * earlier resident path is gone (only the final logits are read back).
 *
 * <p>Holds the full history; the local/global window is a read-time mask (see
 * {@link Gemma3AttentionLayout#firstValidKey}), not eviction. {@link #length()} tracks committed positions
 * exactly like the host {@link Gemma3WarpKvCache}; {@link #reset()} restarts a sequence (buffers are reused
 * and overwritten). Requires the GPU resident kernels.</p>
 */
public final class Gemma3WarpResidentKvCache implements AutoCloseable {

    private final Gemma3WarpResidentKvLayerCache[] layers;
    private final int kvDim;
    private int length;
    private boolean closed;

    public Gemma3WarpResidentKvCache(WarpExecutionContext ctx, int numLayers, int kvDim, int initialCapacity)
            throws WindowsNativeException {
        if (numLayers < 1 || kvDim < 1) {
            throw new IllegalArgumentException("numLayers and kvDim must be positive");
        }
        this.kvDim = kvDim;
        this.layers = new Gemma3WarpResidentKvLayerCache[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new Gemma3WarpResidentKvLayerCache(ctx, kvDim, initialCapacity);
        }
    }

    public int length() {
        return length;
    }

    public int kvDim() {
        return kvDim;
    }

    public int numLayers() {
        return layers.length;
    }

    /** Mark {@code newLength} positions as committed (once a token was appended to all layers). */
    public void commitLength(int newLength) {
        this.length = newLength;
    }

    public void reset() {
        this.length = 0;
    }

    /**
     * Ensure every layer can hold {@code neededPositions}. Call <b>before</b> opening the per-layer decode
     * batch — growth reallocates + copies synchronously and must not run while deferred work references the
     * old buffers.
     */
    public void ensureCapacity(WarpExecutionContext ctx, int neededPositions) throws WindowsNativeException {
        for (Gemma3WarpResidentKvLayerCache layer : layers) {
            layer.ensureCapacity(ctx, neededPositions, length);
        }
    }

    Gemma3WarpResidentKvLayerCache layer(int index) {
        return layers[index];
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            for (Gemma3WarpResidentKvLayerCache layer : layers) {
                layer.close();
            }
        }
    }
}
