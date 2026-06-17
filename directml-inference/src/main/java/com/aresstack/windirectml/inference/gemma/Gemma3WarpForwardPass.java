package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.inference.decoderonly.DecoderOnlyMath;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.util.Objects;

/**
 * Full Gemma 3 prompt prefill on WARP/DirectML (GEMMA-WARP-9): {@code token ids → embedding ×
 * sqrt(hidden) → 18 Gemma WARP layers → final zero-centered RMSNorm → tied LM head → logits for the
 * last position}.
 *
 * <p>Composes the validated building blocks: {@link Gemma3WarpEmbedding} (lookup ×sqrt(hidden)), one
 * {@link Gemma3WarpLayer} per layer (per-layer theta/mask from {@link Gemma3AttentionLayout}), a
 * {@link Gemma3WarpRmsNormKernel} for the final norm, and the tied {@link Gemma3WarpLmHead}. The
 * embedding doubles as the LM head (tied), uploaded once. Not the product decode session yet (no KV
 * cache, no streaming). Requires {@link WindowsBindings#isSupported()}.</p>
 */
public final class Gemma3WarpForwardPass implements AutoCloseable {

    private final Gemma3WarpWeights weights;
    private final Gemma3Config config;
    private final int hidden;
    private final float eps;
    private final float embeddingScale;

    private final WindowsBindings wb;
    private final Gemma3WarpKernels kernels;
    private final Gemma3WarpLayer[] layers;
    private final Gemma3WarpRmsNormKernel finalNorm;
    private Gemma3WarpLmHead lmHead; // lazily built on first logits() call (9a needs no LM head)
    private boolean closed;

    public Gemma3WarpForwardPass(WindowsBindings wb, Gemma3WarpWeights weights) throws WindowsNativeException {
        Objects.requireNonNull(wb, "wb");
        this.wb = wb;
        this.weights = Objects.requireNonNull(weights, "weights");
        this.config = weights.config();
        this.hidden = config.hiddenSize();
        this.eps = (float) config.rmsNormEps();
        this.embeddingScale = (float) config.embeddingScale();

        this.kernels = new Gemma3WarpKernels(wb);
        this.layers = new Gemma3WarpLayer[config.numHiddenLayers()];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Gemma3WarpLayer(wb, config, i, weights.layers()[i], kernels);
        }
        this.finalNorm = new Gemma3WarpRmsNormKernel(wb);
    }

    private Gemma3WarpLmHead lmHead() {
        if (lmHead == null) {
            lmHead = weights.buildLmHead(wb);
        }
        return lmHead;
    }

    /** The post-final-norm hidden vector for the last position (GEMMA-WARP-9a, no LM head). */
    public float[] finalHiddenForLastToken(int[] tokenIds) throws WindowsNativeException {
        ensureOpen();
        if (tokenIds == null || tokenIds.length == 0) {
            throw new IllegalArgumentException("tokenIds must not be empty");
        }
        float[][] state = weights.embedScaled(tokenIds, embeddingScale);
        for (Gemma3WarpLayer layer : layers) {
            layer.forward(state);
        }
        return finalNorm.normalize(state[state.length - 1], weights.finalNorm(), eps);
    }

    /** Vocab-sized logits for the last position (GEMMA-WARP-9b, tied LM head). */
    public float[] logitsForLastToken(int[] tokenIds) throws WindowsNativeException {
        return lmHead().logits(finalHiddenForLastToken(tokenIds));
    }

    /** Greedy next token id for the last position. */
    public int nextToken(int[] tokenIds) throws WindowsNativeException {
        return DecoderOnlyMath.argmax(logitsForLastToken(tokenIds));
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Gemma3WarpForwardPass is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            for (Gemma3WarpLayer layer : layers) {
                layer.close();
            }
            finalNorm.close();
            kernels.close();
            if (lmHead != null) {
                lmHead.close();
            }
        }
    }
}
