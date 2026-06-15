package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.warp.WarpDenseProjection;
import com.aresstack.windirectml.windows.WindowsBindings;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * Gemma 3 tied LM head for the WARP prefill (GEMMA-WARP-9b): {@code logits = hidden · embed_tokens^T}.
 *
 * <p>Gemma ties the LM head to {@code embed_tokens} ({@code [vocab, hidden]}), so there is no separate
 * {@code lm_head} tensor — this projection reuses the embedding matrix. It is the heap/perf-critical
 * piece (vocab ≈ 256k, hidden = 640 → ≈ 164M weights). The matrix is uploaded to the GPU <b>once</b>
 * via {@link WarpDenseProjection}; prefer {@link #fromFp32ByteBuffer} (heap-light: a row-major
 * little-endian FP32 {@link ByteBuffer}, e.g. a decoded SafeTensors payload, no host {@code float[]}
 * copy) over {@link #fromFloatArray} (reference/synthetic convenience). Requires
 * {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpLmHead implements AutoCloseable {

    private final int vocab;
    private final int hidden;
    private final WarpDenseProjection projection;
    private boolean closed;

    private Gemma3WarpLmHead(int vocab, int hidden, WarpDenseProjection projection) {
        this.vocab = vocab;
        this.hidden = hidden;
        this.projection = projection;
    }

    /** Heap-light: build from a row-major {@code [vocab, hidden]} little-endian FP32 {@link ByteBuffer}. */
    public static Gemma3WarpLmHead fromFp32ByteBuffer(WindowsBindings wb, int vocab, int hidden, ByteBuffer fp32Le) {
        Objects.requireNonNull(fp32Le, "fp32Le");
        WarpDenseProjection p = WarpDenseProjection.fromDequantizedWeights(wb, "gemma3.lm_head", vocab, hidden, fp32Le);
        return new Gemma3WarpLmHead(vocab, hidden, p);
    }

    /** Build from a row-major {@code [vocab, hidden]} {@code float[]} (reference/synthetic). */
    public static Gemma3WarpLmHead fromFloatArray(WindowsBindings wb, int vocab, int hidden, float[] embedding) {
        Objects.requireNonNull(embedding, "embedding");
        WarpDenseProjection p = WarpDenseProjection.fromDequantizedWeights(wb, "gemma3.lm_head", vocab, hidden, embedding);
        return new Gemma3WarpLmHead(vocab, hidden, p);
    }

    public int vocab() {
        return vocab;
    }

    /** Vocab-sized logits for a single {@code hidden}-wide (already final-normed) vector. */
    public float[] logits(float[] hiddenState) {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpLmHead is closed");
        }
        Objects.requireNonNull(hiddenState, "hiddenState");
        if (hiddenState.length != hidden) {
            throw new IllegalArgumentException("hidden length mismatch: " + hiddenState.length + " != " + hidden);
        }
        return projection.project(hiddenState);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            projection.close();
        }
    }
}
