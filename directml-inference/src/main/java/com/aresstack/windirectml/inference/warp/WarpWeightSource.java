package com.aresstack.windirectml.inference.warp;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Family-neutral source for a row-major {@code [outputRows, inputColumns]} dense weight matrix uploaded to a WARP
 * dense projection.
 *
 * <p>This is the single shared contract behind the heap-light WARP upload (slices H2b/H2c): a model family describes a
 * weight as either a raw little-endian FP32 {@link ByteBuffer} slice (preferred — uploaded directly, no host
 * {@code float[]}) <em>or</em> a {@code float[]} fallback supplier (FP16/BF16, fused/reference tensors). The decision
 * "ByteBuffer if present, else float[]" lives once in {@link WarpDenseProjection#fromWeightSource} instead of being
 * duplicated per family (T5, SmolLM2, …).</p>
 *
 * <p>The {@code float[]} fallback is a {@link Supplier} so it is only invoked when the FP32 ByteBuffer is absent — a
 * present ByteBuffer never forces a host {@code float[]} materialisation. No runtime format is changed.</p>
 */
public final class WarpWeightSource {

    private final String name;
    private final int outputRows;
    private final int inputColumns;
    private final ByteBuffer fp32LittleEndian;        // nullable (a retained FP32 slice)
    private final Supplier<ByteBuffer> fp32LeSupplier; // nullable (GEMMA-BF16-PACK-3: lazy/transient FP32 widen)
    private final Supplier<float[]> dequantizedFallback;

    private WarpWeightSource(String name, int outputRows, int inputColumns, ByteBuffer fp32LittleEndian,
                             Supplier<ByteBuffer> fp32LeSupplier, Supplier<float[]> dequantizedFallback) {
        this.name = Objects.requireNonNull(name, "name");
        this.outputRows = outputRows;
        this.inputColumns = inputColumns;
        this.fp32LittleEndian = fp32LittleEndian;
        this.fp32LeSupplier = fp32LeSupplier;
        this.dequantizedFallback = Objects.requireNonNull(dequantizedFallback, "dequantizedFallback");
    }

    /**
     * A source that prefers the {@code fp32LittleEndian} slice (when non-null) and otherwise falls back to
     * {@code dequantizedFallback} (only invoked then). This is the common adapter for family tensor views whose
     * FP32 source may or may not be available (T5TensorData / SmolLM2DenseTensor).
     */
    public static WarpWeightSource of(String name, int outputRows, int inputColumns,
                                      ByteBuffer fp32LittleEndian, Supplier<float[]> dequantizedFallback) {
        return new WarpWeightSource(name, outputRows, inputColumns, fp32LittleEndian, null, dequantizedFallback);
    }

    /**
     * A source whose FP32 little-endian bytes are produced <b>lazily and transiently</b> by {@code supplier}
     * (GEMMA-BF16-PACK-3) — for a retained BF16 weight that is widened to FP32 only for the device upload/
     * prepacking, not kept as FP32. The supplier is invoked at most once per projection build. There is no
     * {@code float[]} fallback (it would defeat the heap-light intent).
     */
    public static WarpWeightSource ofLazyFp32(String name, int outputRows, int inputColumns,
                                              Supplier<ByteBuffer> supplier) {
        Objects.requireNonNull(supplier, "supplier");
        return new WarpWeightSource(name, outputRows, inputColumns, null, supplier,
                () -> {
                    throw new IllegalStateException("no float[] fallback for a lazy-FP32 source: " + name);
                });
    }

    public String name() {
        return name;
    }

    public int outputRows() {
        return outputRows;
    }

    public int inputColumns() {
        return inputColumns;
    }

    /**
     * Whether a direct FP32 little-endian upload slice is available (no host {@code float[]} needed).
     */
    public boolean hasFp32LittleEndian() {
        return fp32LittleEndian != null;
    }

    /**
     * The raw little-endian FP32 upload slice, or {@code null}. Returns an independent view when present.
     * (Retained-only; does not invoke a lazy supplier — used by callers that have not adopted lazy FP32.)
     */
    public ByteBuffer fp32LittleEndian() {
        return fp32LittleEndian == null ? null : fp32LittleEndian.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * An upload-ready little-endian FP32 buffer, or {@code null} if this source only has a {@code float[]}
     * fallback (GEMMA-BF16-PACK-3): the retained slice when present, otherwise the lazy supplier's transient
     * widen. Prefer this over {@link #fp32LittleEndian()} when a source may be BF16-backed.
     */
    public ByteBuffer resolveFp32LittleEndian() {
        if (fp32LittleEndian != null) {
            return fp32LittleEndian.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        }
        if (fp32LeSupplier != null) {
            ByteBuffer b = fp32LeSupplier.get();
            return b == null ? null : b.order(ByteOrder.LITTLE_ENDIAN);
        }
        return null;
    }

    /**
     * The {@code float[]} fallback. Only call this when {@link #hasFp32LittleEndian()} is false.
     */
    public float[] dequantizedRowMajor() {
        return dequantizedFallback.get();
    }
}
