package com.aresstack.windirectml.inference.phi3;

import java.nio.file.Path;

/**
 * Heap-light compile layout for the Phi-3 wdmlpack compiler (PHI3-WDMLPACK-COMPILER-2).
 *
 * <p>Describes <em>where</em> each role tensor lives in the ONNX source (external-data offset/length/dtype, or a small
 * already-converted inline vector) without materializing the ~2.4 GB of weights on the heap. The streaming compiler
 * then copies/converts each tensor directly from the mmap'd {@code model.onnx.data} into the {@code .wdmlpack}, so the
 * compile peak is a few bounded buffers instead of a full {@link Phi3Weights}.</p>
 */
public final class Phi3WeightLayout {

    /** A tensor stored in the external {@code model.onnx.data} file. */
    public record ExtRef(long offset, long length, int onnxDtype, long[] dims) {
    }

    /** INT4 MatMulNBits triplet. {@code zeroPoints} may be {@code null} (default zp=8 fill is generated). */
    public record QuantRef(ExtRef qData, ExtRef scales, ExtRef zeroPoints, int N, int K, int blockSize) {
    }

    /** Per-layer source refs (norms/scales small + 6 quantized projections). */
    public record LayerRef(ExtRef inputNorm, QuantRef qProj, QuantRef kProj, QuantRef vProj,
                           float[] attnOutScale, QuantRef oProj, ExtRef postNorm,
                           QuantRef gateUpProj, float[] mlpOutScale, QuantRef downProj) {
    }

    private final Phi3Config config;
    private final Path externalDataPath;
    private final ExtRef embedTokens;
    private final float[] cosCache;
    private final float[] sinCache;
    private final ExtRef finalNorm;
    private final QuantRef lmHead;
    private final LayerRef[] layers;

    Phi3WeightLayout(Phi3Config config, Path externalDataPath, ExtRef embedTokens, float[] cosCache,
                     float[] sinCache, ExtRef finalNorm, QuantRef lmHead, LayerRef[] layers) {
        this.config = config;
        this.externalDataPath = externalDataPath;
        this.embedTokens = embedTokens;
        this.cosCache = cosCache;
        this.sinCache = sinCache;
        this.finalNorm = finalNorm;
        this.lmHead = lmHead;
        this.layers = layers;
    }

    public Phi3Config config() {
        return config;
    }

    public Path externalDataPath() {
        return externalDataPath;
    }

    public ExtRef embedTokens() {
        return embedTokens;
    }

    public float[] cosCache() {
        return cosCache;
    }

    public float[] sinCache() {
        return sinCache;
    }

    public ExtRef finalNorm() {
        return finalNorm;
    }

    public QuantRef lmHead() {
        return lmHead;
    }

    public LayerRef[] layers() {
        return layers;
    }
}
