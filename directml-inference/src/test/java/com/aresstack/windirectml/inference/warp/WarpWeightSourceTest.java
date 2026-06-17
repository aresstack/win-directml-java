package com.aresstack.windirectml.inference.warp;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Device-free behaviour of the shared {@link WarpWeightSource} contract used by T5 and SmolLM2 (todo item 2).
 */
class WarpWeightSourceTest {

    @Test
    void fp32SourcePrefersByteBufferAndExposesIndependentViews() {
        ByteBuffer le = ByteBuffer.allocate(4 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        AtomicInteger fallbackCalls = new AtomicInteger();
        WarpWeightSource src = WarpWeightSource.of("w", 2, 2, le, () -> {
            fallbackCalls.incrementAndGet();
            return new float[]{1, 2, 3, 4};
        });

        assertTrue(src.hasFp32LittleEndian());
        assertEquals(2, src.outputRows());
        assertEquals(2, src.inputColumns());

        ByteBuffer a = src.fp32LittleEndian();
        a.position(8);
        ByteBuffer b = src.fp32LittleEndian();
        assertEquals(0, b.position(), "each fp32LittleEndian() call must return an independent view");
        assertEquals(ByteOrder.LITTLE_ENDIAN, b.order());
        assertEquals(0, fallbackCalls.get(), "fallback supplier must not be touched when a ByteBuffer is present");
    }

    @Test
    void fallbackSourceUsesSupplierWhenNoByteBuffer() {
        AtomicInteger fallbackCalls = new AtomicInteger();
        WarpWeightSource src = WarpWeightSource.of("w", 1, 4, null, () -> {
            fallbackCalls.incrementAndGet();
            return new float[]{5, 6, 7, 8};
        });

        assertFalse(src.hasFp32LittleEndian());
        assertNull(src.fp32LittleEndian());
        assertArrayEquals(new float[]{5, 6, 7, 8}, src.dequantizedRowMajor(), 0.0f);
        assertEquals(1, fallbackCalls.get());
    }

    // ── GEMMA-BF16-PACK-3: lazy/transient FP32 source ──────────────────────────────────────

    @Test
    void retainedFp32IsReturnedByResolve() {
        ByteBuffer le = ByteBuffer.allocate(4 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        WarpWeightSource src = WarpWeightSource.of("w", 2, 2, le, () -> {
            throw new AssertionError("fallback must not run");
        });
        ByteBuffer resolved = src.resolveFp32LittleEndian();
        assertEquals(4 * Float.BYTES, resolved.remaining());
        assertEquals(ByteOrder.LITTLE_ENDIAN, resolved.order());
    }

    @Test
    void floatOnlySourceResolvesToNull() {
        WarpWeightSource src = WarpWeightSource.of("w", 1, 4, null, () -> new float[]{1, 2, 3, 4});
        assertNull(src.resolveFp32LittleEndian(), "no FP32 source -> caller uses the float[] path");
    }

    @Test
    void lazyFp32SourceInvokesSupplierOnlyOnResolve() {
        ByteBuffer transient32 = ByteBuffer.allocate(2 * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        AtomicInteger calls = new AtomicInteger();
        WarpWeightSource src = WarpWeightSource.ofLazyFp32("w", 2, 1, () -> {
            calls.incrementAndGet();
            return transient32;
        });
        assertNull(src.fp32LittleEndian(), "lazy source has no retained FP32 slice");
        assertEquals(0, calls.get(), "supplier must not run before resolve");
        ByteBuffer resolved = src.resolveFp32LittleEndian();
        assertEquals(ByteOrder.LITTLE_ENDIAN, resolved.order());
        assertEquals(1, calls.get(), "supplier runs once on resolve");
    }
}
