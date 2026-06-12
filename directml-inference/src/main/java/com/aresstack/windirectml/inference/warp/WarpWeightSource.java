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
    private final ByteBuffer fp32LittleEndian;        // nullable
    private final Supplier<float[]> dequantizedFallback;

    private WarpWeightSource(String name, int outputRows, int inputColumns,
                             ByteBuffer fp32LittleEndian, Supplier<float[]> dequantizedFallback) {
        this.name = Objects.requireNonNull(name, "name");
        this.outputRows = outputRows;
        this.inputColumns = inputColumns;
        this.fp32LittleEndian = fp32LittleEndian;
        this.dequantizedFallback = Objects.requireNonNull(dequantizedFallback, "dequantizedFallback");
    }

    /**
     * A source that prefers the {@code fp32LittleEndian} slice (when non-null) and otherwise falls back to
     * {@code dequantizedFallback} (only invoked then). This is the common adapter for family tensor views whose
     * FP32 source may or may not be available (T5TensorData / SmolLM2DenseTensor).
     */
    public static WarpWeightSource of(String name, int outputRows, int inputColumns,
                                      ByteBuffer fp32LittleEndian, Supplier<float[]> dequantizedFallback) {
        return new WarpWeightSource(name, outputRows, inputColumns, fp32LittleEndian, dequantizedFallback);
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
     */
    public ByteBuffer fp32LittleEndian() {
        return fp32LittleEndian == null ? null : fp32LittleEndian.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * The {@code float[]} fallback. Only call this when {@link #hasFp32LittleEndian()} is false.
     */
    public float[] dequantizedRowMajor() {
        return dequantizedFallback.get();
    }
}
