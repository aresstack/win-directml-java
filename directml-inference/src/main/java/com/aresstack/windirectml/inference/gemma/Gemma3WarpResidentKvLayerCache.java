package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WarpExecutionContext;
import com.aresstack.windirectml.windows.WarpGpuBuffer;
import com.aresstack.windirectml.windows.WindowsNativeException;

/**
 * One layer's GPU-resident key/value cache for the Gemma 3 WARP decode session (GEMMA-WARP-13c): two
 * device buffers ({@code K}, {@code V}) of {@code capacity × kvDim} floats that the decode step appends to
 * (via the UAV append kernel) and the attention kernels read directly — no per-token CPU readback/upload.
 *
 * <p>Stores the <b>full</b> history (row-major {@code [position][kvDim]}); the local/global sliding-window
 * restriction is applied at read time by {@link Gemma3AttentionLayout#firstValidKey} (mask), not by
 * evicting rows — windowed eviction is a later perf concern. Growth (rare) reallocates and copies the
 * valid prefix; it must run outside a submission batch so the old buffer can be freed safely.</p>
 */
final class Gemma3WarpResidentKvLayerCache implements AutoCloseable {

    private final int kvDim;
    private WarpGpuBuffer k;
    private WarpGpuBuffer v;
    private int capacity; // positions
    private boolean closed;

    Gemma3WarpResidentKvLayerCache(WarpExecutionContext ctx, int kvDim, int initialCapacityPositions)
            throws WindowsNativeException {
        this.kvDim = kvDim;
        this.capacity = Math.max(1, initialCapacityPositions);
        this.k = ctx.allocate(this.capacity * kvDim);
        this.v = ctx.allocate(this.capacity * kvDim);
    }

    /**
     * Ensure room for {@code neededPositions}; grows by doubling and copies the {@code validPositions}
     * prefix into the new buffers. Must be called outside a submission batch (the copy is synchronous so
     * the old buffers can be freed immediately).
     */
    void ensureCapacity(WarpExecutionContext ctx, int neededPositions, int validPositions)
            throws WindowsNativeException {
        if (neededPositions <= capacity) {
            return;
        }
        int newCap = capacity;
        while (newCap < neededPositions) {
            newCap *= 2;
        }
        WarpGpuBuffer newK = ctx.allocate(newCap * kvDim);
        WarpGpuBuffer newV = ctx.allocate(newCap * kvDim);
        if (validPositions > 0) {
            ctx.copyRegionInto(newK, 0, k, 0, validPositions * kvDim);
            ctx.copyRegionInto(newV, 0, v, 0, validPositions * kvDim);
        }
        k.close();
        v.close();
        k = newK;
        v = newV;
        capacity = newCap;
    }

    WarpGpuBuffer keys() {
        return k;
    }

    WarpGpuBuffer values() {
        return v;
    }

    int capacityPositions() {
        return capacity;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            k.close();
            v.close();
        }
    }
}
