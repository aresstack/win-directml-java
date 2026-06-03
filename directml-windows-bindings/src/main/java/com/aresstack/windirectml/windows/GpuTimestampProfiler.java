package com.aresstack.windirectml.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

/**
 * Reusable D3D12 timestamp-query profiler for one command-list submission.
 */
public final class GpuTimestampProfiler implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(GpuTimestampProfiler.class);

    private final Arena arena;
    private final MemorySegment queryHeap;
    private final MemorySegment readbackBuffer;
    private final MemorySegment mappedReadback;
    private final long frequency;
    private final int maxQueries;
    private final List<String> labels = new ArrayList<>();

    private int queryIndex;
    private boolean resolved;
    private boolean closed;

    private GpuTimestampProfiler(Arena arena, MemorySegment queryHeap, MemorySegment readbackBuffer,
                                 MemorySegment mappedReadback, long frequency, int maxQueries) {
        this.arena = arena;
        this.queryHeap = queryHeap;
        this.readbackBuffer = readbackBuffer;
        this.mappedReadback = mappedReadback;
        this.frequency = frequency;
        this.maxQueries = maxQueries;
    }

    public static GpuTimestampProfiler create(WindowsBindings bindings, int maxQueries)
            throws WindowsNativeException {
        if (maxQueries < 2) {
            throw new IllegalArgumentException("maxQueries must be at least 2");
        }
        Arena arena = Arena.ofShared();
        try {
            MemorySegment queryHeap = D3D12Bindings.createTimestampQueryHeap(
                    bindings.getD3d12Device(), maxQueries, arena);
            MemorySegment readbackBuffer = D3D12Bindings.createReadbackBuffer(
                    bindings.getD3d12Device(), (long) maxQueries * Long.BYTES, arena);
            MemorySegment mappedReadback = D3D12Bindings.mapResource(readbackBuffer, arena);
            long frequency = D3D12Bindings.getTimestampFrequency(bindings.getCommandQueue(), arena);
            return new GpuTimestampProfiler(arena, queryHeap, readbackBuffer,
                    mappedReadback, frequency, maxQueries);
        } catch (WindowsNativeException | RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    public static GpuTimestampProfiler tryCreate(WindowsBindings bindings, int maxQueries) {
        try {
            GpuTimestampProfiler profiler = create(bindings, maxQueries);
            log.info("D3D12 timestamp profiler enabled: maxQueries={}, frequency={} Hz",
                    maxQueries, profiler.frequency);
            return profiler;
        } catch (Exception e) {
            log.warn("D3D12 timestamp profiler unavailable: {}", e.getMessage());
            return null;
        }
    }

    public void reset() {
        ensureOpen();
        labels.clear();
        queryIndex = 0;
        resolved = false;
    }

    public void record(MemorySegment commandList, String label) {
        ensureOpen();
        if (resolved) {
            throw new IllegalStateException("Timestamp queries have already been resolved");
        }
        if (queryIndex >= maxQueries) {
            throw new IllegalStateException("Timestamp query capacity exceeded: " + maxQueries);
        }
        labels.add(label);
        D3D12Bindings.endTimestampQuery(commandList, queryHeap, queryIndex);
        queryIndex++;
    }

    public void resolve(MemorySegment commandList) {
        ensureOpen();
        if (queryIndex == 0) {
            return;
        }
        D3D12Bindings.resolveTimestampQueries(commandList, queryHeap, 0, queryIndex, readbackBuffer, 0L);
        resolved = true;
    }

    public GpuTimestampProfile readProfile() {
        ensureOpen();
        if (!resolved || queryIndex == 0) {
            return new GpuTimestampProfile(frequency, java.util.Collections.emptyList(), new long[0]);
        }

        long[] ticks = new long[queryIndex];
        for (int index = 0; index < queryIndex; index++) {
            ticks[index] = mappedReadback.get(ValueLayout.JAVA_LONG, (long) index * Long.BYTES);
        }
        return new GpuTimestampProfile(frequency, labels, ticks);
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            D3D12Bindings.unmapResource(readbackBuffer);
        } catch (Exception e) {
            log.debug("Timestamp readback unmap failed: {}", e.getMessage());
        }
        DxgiBindings.release(readbackBuffer);
        DxgiBindings.release(queryHeap);
        arena.close();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GpuTimestampProfiler is closed");
        }
    }
}
