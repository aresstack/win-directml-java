package com.aresstack.windirectml.encoder.e5;

import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.aresstack.windirectml.encoder.bert.BertCpuEncoderWeights;
import com.aresstack.windirectml.encoder.bert.BertCpuLayerWeights;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.encoder.bert.CpuBertEncoder;
import com.aresstack.windirectml.encoder.bert.DirectMlBertEncoder;
import com.aresstack.windirectml.encoder.util.CosineSimilarity;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Synthetic-weight parity test for the E5 family.
 * <p>
 * Builds a tiny E5-shaped {@link BertEncoderConfig} (deliberately
 * different from MiniLM: 2 layers, hidden 24, intermediate 48,
 * heads 4) with random weights, runs the same input through
 * {@link CpuBertEncoder} and {@link DirectMlBertEncoder}, and
 * asserts the two outputs match by cosine similarity.
 * <p>
 * Exercises the generic {@code encoder.bert} pipeline with a
 * non-MiniLM shape – validates that adding a second encoder family
 * to the runtime really did not require any DirectML kernel changes.
 */
@EnabledOnOs(OS.WINDOWS)
class E5SyntheticParityTest {

    /** Tiny WordPiece-shaped tokenizer matching the synthetic vocab. */
    private static final class TinyTokenizer implements EncoderTokenizer {
        private final int vocab;
        TinyTokenizer(int vocab) { this.vocab = vocab; }
        @Override public Encoded encode(String text) {
            int n = Math.min(text.length() + 2, 8);
            int[] ids = new int[n];
            int[] mask = new int[n];
            int[] segs = new int[n];
            ids[0] = 2;                                  // CLS
            for (int i = 1; i < n - 1; i++) ids[i] = 4 + (text.charAt(i - 1) % (vocab - 5));
            ids[n - 1] = 3;                              // SEP
            for (int i = 0; i < n; i++) mask[i] = 1;
            return new Encoded(ids, mask, segs);
        }
        @Override public int padTokenId() { return 0; }
        @Override public int clsTokenId() { return 2; }
        @Override public int sepTokenId() { return 3; }
        @Override public int vocabSize()  { return vocab; }
    }

    private static BertEncoderConfig tinyE5Config() {
        return new BertEncoderConfig(
                "test/e5-tiny",
                /* hidden            */ 24,
                /* numLayers         */ 2,
                /* numHeads          */ 4,
                /* intermediate      */ 48,
                /* maxPos            */ 16,
                /* typeVocab         */ 2,
                /* vocab             */ 32,
                /* layerNormEps      */ 1e-12f,
                /* hiddenAct         */ "gelu",
                /* outputDimension   */ 24,
                /* pooling           */ PoolingStrategy.MEAN,
                /* normalize         */ true);
    }

    private static BertCpuEncoderWeights randomWeights(BertEncoderConfig cfg, long seed) {
        Random r = new Random(seed);
        int H = cfg.hiddenSize(), I = cfg.intermediateSize();
        float[] we  = rand(r, cfg.vocabSize() * H, 0.02f);
        float[] pe  = rand(r, cfg.maxPositionEmbeddings() * H, 0.02f);
        float[] tte = rand(r, cfg.typeVocabSize() * H, 0.02f);
        float[] elng = ones(H);
        float[] elnb = zeros(H);
        List<BertCpuLayerWeights> layers = new ArrayList<>();
        for (int l = 0; l < cfg.numLayers(); l++) {
            layers.add(new BertCpuLayerWeights(
                    rand(r, H * H, 0.05f), zeros(H),
                    rand(r, H * H, 0.05f), zeros(H),
                    rand(r, H * H, 0.05f), zeros(H),
                    rand(r, H * H, 0.05f), zeros(H),
                    ones(H), zeros(H),
                    rand(r, I * H, 0.05f), zeros(I),
                    rand(r, H * I, 0.05f), zeros(H),
                    ones(H), zeros(H)));
        }
        return BertCpuEncoderWeights.forTesting(cfg, we, pe, tte, elng, elnb, layers);
    }

    @Test
    void cpuAndDirectMlAgreeOnSyntheticE5Shape() throws Exception {
        assumeTrue(WindowsBindings.isSupported(),
                "DirectML requires Windows + D3D12 on this host");
        BertEncoderConfig cfg = tinyE5Config();
        BertCpuEncoderWeights w = randomWeights(cfg, 4242L);
        TinyTokenizer tok = new TinyTokenizer(cfg.vocabSize());

        try (DirectMlContextImpl ctx = new DirectMlContextImpl("e5-parity")) {
            ctx.initialize();
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                    "No DirectML device available on this adapter");

            try (CpuBertEncoder cpu = new CpuBertEncoder(cfg, w, tok);
                 DirectMlBertEncoder gpu = DirectMlBertEncoder.build(ctx, /* ownsCtx */ false,
                         cfg, w, tok)) {

                String text = "the quick brown fox jumps";
                EmbeddingVector vCpu = cpu.embed(new EmbeddingRequest(text, true, E5Prefixes.QUERY));
                EmbeddingVector vGpu = gpu.embed(new EmbeddingRequest(text, true, E5Prefixes.QUERY));

                assertEquals(cfg.outputDimension(), vCpu.dimension());
                assertEquals(cfg.outputDimension(), vGpu.dimension());

                double cos = CosineSimilarity.compute(vCpu.values(), vGpu.values());
                assertTrue(cos > 0.999,
                        "CPU↔DirectML cosine on synthetic E5 must be > 0.999, was " + cos);

                // Both must be unit-norm (normalize=true was passed)
                assertTrue(Math.abs(norm(vCpu.values()) - 1.0) < 1e-4,
                        "CPU E5 output must be unit-norm");
                assertTrue(Math.abs(norm(vGpu.values()) - 1.0) < 1e-4,
                        "DirectML E5 output must be unit-norm");
            }
        }
    }

    @Test
    void prefixesAreAppliedBeforeTokenisation() throws Exception {
        assumeTrue(WindowsBindings.isSupported(),
                "DirectML requires Windows + D3D12 on this host");
        BertEncoderConfig cfg = tinyE5Config();
        BertCpuEncoderWeights w = randomWeights(cfg, 7L);
        TinyTokenizer tok = new TinyTokenizer(cfg.vocabSize());

        try (CpuBertEncoder cpu = new CpuBertEncoder(cfg, w, tok)) {
            // Same text, different roles → different vectors (because the
            // tokenized prefix changes the input sequence). This proves
            // the EmbeddingRequest.prefix() really flows through.
            EmbeddingVector q = cpu.embed(E5Prefixes.request("german engineering", E5Prefixes.Role.QUERY, true));
            EmbeddingVector p = cpu.embed(E5Prefixes.request("german engineering", E5Prefixes.Role.PASSAGE, true));
            double cos = CosineSimilarity.compute(q.values(), p.values());
            // They are not identical (different prefix → different tokens),
            // but with random weights they should also not be wildly orthogonal.
            assertTrue(cos < 0.9999,
                    "query/passage prefixes must change the output (cos=" + cos + ")");
        }
    }

    private static float[] rand(Random r, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) (r.nextGaussian() * scale);
        return a;
    }
    private static float[] ones(int n) { float[] a = new float[n]; java.util.Arrays.fill(a, 1f); return a; }
    private static float[] zeros(int n) { return new float[n]; }
    private static double norm(float[] v) {
        double s = 0; for (float x : v) s += (double) x * x; return Math.sqrt(s);
    }
}

