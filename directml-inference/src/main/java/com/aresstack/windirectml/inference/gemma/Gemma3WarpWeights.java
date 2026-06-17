package com.aresstack.windirectml.inference.gemma;

import com.aresstack.windirectml.windows.WindowsBindings;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Whole-model Gemma 3 weights for the WARP prefill (GEMMA-WARP-9): the tied {@code embed_tokens}
 * matrix, the final norm, and the per-layer {@link Gemma3WarpLayerWeights}.
 *
 * <p>The embedding is held as exactly one of: a host {@code float[]} (reference/synthetic) or a
 * row-major little-endian FP32 {@link ByteBuffer} (heap-light, e.g. a decoded SafeTensors payload). It
 * is the tied LM head too — one matrix, materialised/uploaded once.</p>
 */
public final class Gemma3WarpWeights {

    private final Gemma3Config config;
    private final float[] embeddingFloat;        // nullable
    private final ByteBuffer embeddingFp32Le;    // nullable
    private final Gemma3Bf16WeightView embeddingBf16; // nullable (GEMMA-BF16-PACK-2: retained BF16 tied block)
    private final float[] finalNorm;
    private final Gemma3WarpLayerWeights[] layers;

    private Gemma3WarpWeights(Gemma3Config config, float[] embeddingFloat, ByteBuffer embeddingFp32Le,
                             Gemma3Bf16WeightView embeddingBf16, float[] finalNorm,
                             Gemma3WarpLayerWeights[] layers) {
        this.config = Objects.requireNonNull(config, "config");
        this.embeddingFloat = embeddingFloat;
        this.embeddingFp32Le = embeddingFp32Le;
        this.embeddingBf16 = embeddingBf16;
        this.finalNorm = Objects.requireNonNull(finalNorm, "finalNorm");
        this.layers = Objects.requireNonNull(layers, "layers");
        if (layers.length != config.numHiddenLayers()) {
            throw new IllegalArgumentException("layers length must equal numHiddenLayers: " + layers.length
                    + " != " + config.numHiddenLayers());
        }
    }

    /** Build with a host {@code float[]} embedding ({@code [vocab, hidden]} row-major). */
    public static Gemma3WarpWeights ofFloatEmbedding(Gemma3Config config, float[] embedding,
                                                     float[] finalNorm, Gemma3WarpLayerWeights[] layers) {
        return new Gemma3WarpWeights(config, Objects.requireNonNull(embedding, "embedding"), null, null,
                finalNorm, layers);
    }

    /** Build heap-light with a row-major {@code [vocab, hidden]} little-endian FP32 {@link ByteBuffer}. */
    public static Gemma3WarpWeights ofByteBufferEmbedding(Gemma3Config config, ByteBuffer embeddingFp32Le,
                                                          float[] finalNorm, Gemma3WarpLayerWeights[] layers) {
        Objects.requireNonNull(embeddingFp32Le, "embeddingFp32Le");
        if (embeddingFp32Le.order() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalArgumentException("embedding ByteBuffer must be LITTLE_ENDIAN");
        }
        return new Gemma3WarpWeights(config, null, embeddingFp32Le, null, finalNorm, layers);
    }

    /**
     * Build heap-light with the tied embedding/LM head retained as <b>BF16</b> (GEMMA-BF16-PACK-2): ~half the
     * host RAM of {@link #ofByteBufferEmbedding}. Rows are widened to FP32 on demand for the lookup and once
     * for the LM-head device upload.
     */
    public static Gemma3WarpWeights ofBf16Embedding(Gemma3Config config, Gemma3Bf16WeightView embeddingBf16,
                                                    float[] finalNorm, Gemma3WarpLayerWeights[] layers) {
        Objects.requireNonNull(embeddingBf16, "embeddingBf16");
        return new Gemma3WarpWeights(config, null, null, embeddingBf16, finalNorm, layers);
    }

    /** Build from the loaded CPU reference weights (host {@code float[]} embedding). */
    public static Gemma3WarpWeights from(Gemma3ReferenceWeights ref) {
        Objects.requireNonNull(ref, "ref");
        Gemma3WarpLayerWeights[] wl = new Gemma3WarpLayerWeights[ref.layers.length];
        for (int i = 0; i < wl.length; i++) {
            wl[i] = Gemma3WarpLayerWeights.from(ref.layers[i]);
        }
        return ofFloatEmbedding(ref.config(), ref.embedTokens, ref.finalNorm, wl);
    }

    public Gemma3Config config() {
        return config;
    }

    /**
     * Look up + scale the input embeddings for {@code ids} from whichever representation this holds
     * (GEMMA-BF16-PACK-2): the domain operation "embeddingLookup reads token embeddings", so callers do not
     * branch on the storage form. {@code [ids.length][hidden]}.
     */
    public float[][] embedScaled(int[] ids, float scale) {
        int hidden = config.hiddenSize();
        if (embeddingBf16 != null) {
            return Gemma3WarpEmbedding.lookupScaled(embeddingBf16, ids, hidden, scale);
        }
        if (embeddingFp32Le != null) {
            return Gemma3WarpEmbedding.lookupScaled(embeddingFp32Le, ids, hidden, scale);
        }
        return Gemma3WarpEmbedding.lookupScaled(embeddingFloat, ids, hidden, scale);
    }

    /**
     * Build the tied LM-head projection from whichever embedding representation this holds
     * (GEMMA-BF16-PACK-2): "lmHeadProjection uses the tied embedding weight". The device buffer is FP32
     * regardless of the host storage form.
     */
    public Gemma3WarpLmHead buildLmHead(WindowsBindings wb) {
        int vocab = config.vocabSize();
        int hidden = config.hiddenSize();
        if (embeddingBf16 != null) {
            return Gemma3WarpLmHead.fromBf16View(wb, vocab, hidden, embeddingBf16);
        }
        if (embeddingFp32Le != null) {
            return Gemma3WarpLmHead.fromFp32ByteBuffer(wb, vocab, hidden, embeddingFp32Le);
        }
        return Gemma3WarpLmHead.fromFloatArray(wb, vocab, hidden, embeddingFloat);
    }

    public float[] finalNorm() {
        return finalNorm;
    }

    public Gemma3WarpLayerWeights[] layers() {
        return layers;
    }

    public boolean hasByteBufferEmbedding() {
        return embeddingFp32Le != null;
    }

    public float[] embeddingFloat() {
        return embeddingFloat;
    }

    public ByteBuffer embeddingFp32Le() {
        return embeddingFp32Le;
    }
}
