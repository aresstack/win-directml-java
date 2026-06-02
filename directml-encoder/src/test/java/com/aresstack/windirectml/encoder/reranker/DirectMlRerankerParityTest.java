package com.aresstack.windirectml.encoder.reranker;

import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.aresstack.windirectml.encoder.bert.BertCpuEncoderWeights;
import com.aresstack.windirectml.encoder.bert.BertCpuLayerWeights;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.runtime.DirectMlContextImpl;
import com.aresstack.windirectml.windows.WindowsBindings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Synthetic-weight parity test for the cross-encoder reranker.
 * <p>
 * Builds a tiny BERT-shaped encoder (2 layers, hidden 24, heads 4)
 * plus a {@code [1, H]} classifier head with random weights, runs the
 * exact same {@code (query, document)} pair through {@link CpuReranker}
 * and {@link DirectMlReranker}, and asserts the two scalar scores agree
 * to within float-precision tolerance.
 * <p>
 * Covers acceptance criterion 7 of the reranker sprint ("CPU vs
 * DirectML score parity") and exercises:
 * <ul>
 *   <li>pair tokenisation with {@code token_type_ids};</li>
 *   <li>{@link com.aresstack.windirectml.encoder.bert.DirectMlBertEncoderStack}
 *       on a non-MiniLM shape;</li>
 *   <li>the CPU classification head consuming a GPU-produced
 *       {@code [CLS]} hidden state.</li>
 * </ul>
 */
@EnabledOnOs(OS.WINDOWS)
class DirectMlRerankerParityTest {

    /**
     * Minimal pair-tokenizer: {@code [CLS] a [SEP] b [SEP]} with token-type
     * ids {@code [0…0,1…1]}. Avoids loading a real {@code tokenizer.json}
     * so the test does not depend on disk state.
     */
    private static final class TinyPairTokenizer implements EncoderTokenizer {
        private final int vocab;

        TinyPairTokenizer(int vocab) {
            this.vocab = vocab;
        }

        @Override
        public Encoded encode(String text) {
            return encodePair(text, "");
        }

        @Override
        public Encoded encodePair(String a, String b) {
            int[] aIds = idsFor(a);
            int[] bIds = idsFor(b);
            int n = 1 + aIds.length + 1 + bIds.length + 1;
            int[] ids = new int[n];
            int[] mask = new int[n];
            int[] seg = new int[n];
            int p = 0;
            ids[p++] = 2; // CLS
            for (int x : aIds) {
                ids[p] = x;
                seg[p] = 0;
                p++;
            }
            ids[p] = 3;
            seg[p] = 0;
            p++;       // first SEP
            for (int x : bIds) {
                ids[p] = x;
                seg[p] = 1;
                p++;
            }
            ids[p] = 3;
            seg[p] = 1;            // second SEP
            for (int i = 0; i < n; i++) mask[i] = 1;
            return new Encoded(ids, mask, seg);
        }

        private int[] idsFor(String text) {
            int n = Math.min(text.length(), 6);
            int[] out = new int[n];
            for (int i = 0; i < n; i++) out[i] = 4 + (text.charAt(i) % Math.max(1, vocab - 5));
            return out;
        }

        @Override
        public int padTokenId() {
            return 0;
        }

        @Override
        public int clsTokenId() {
            return 2;
        }

        @Override
        public int sepTokenId() {
            return 3;
        }

        @Override
        public int vocabSize() {
            return vocab;
        }
    }

    private static BertEncoderConfig tinyConfig() {
        return new BertEncoderConfig(
                "test/reranker-tiny",
                /* hidden            */ 24,
                /* numLayers         */ 2,
                /* numHeads          */ 4,
                /* intermediate      */ 48,
                /* maxPos            */ 32,
                /* typeVocab         */ 2,
                /* vocab             */ 32,
                /* layerNormEps      */ 1e-12f,
                /* hiddenAct         */ "gelu",
                /* outputDimension   */ 24,
                /* pooling           */ PoolingStrategy.MEAN,
                /* normalize         */ false);
    }

    private static RerankerCpuWeights tinyWeights(BertEncoderConfig cfg, long seed) {
        Random r = new Random(seed);
        int H = cfg.hiddenSize(), I = cfg.intermediateSize();
        float[] we = rand(r, cfg.vocabSize() * H, 0.02f);
        float[] pe = rand(r, cfg.maxPositionEmbeddings() * H, 0.02f);
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
        BertCpuEncoderWeights bert =
                BertCpuEncoderWeights.forTesting(cfg, we, pe, tte, elng, elnb, layers);
        float[] clsW = rand(r, H, 0.1f);
        float[] clsB = new float[]{(float) (r.nextGaussian() * 0.01)};
        return new RerankerCpuWeights(bert, clsW, clsB, 1);
    }

    @Test
    void cpuAndDirectMlAgreeOnSinglePairScore() throws Exception {
        assumeTrue(WindowsBindings.isSupported(),
                "DirectML requires Windows + D3D12 on this host");
        BertEncoderConfig cfg = tinyConfig();
        RerankerCpuWeights w = tinyWeights(cfg, 9001L);
        TinyPairTokenizer tok = new TinyPairTokenizer(cfg.vocabSize());

        try (DirectMlContextImpl ctx = new DirectMlContextImpl("reranker-parity")) {
            ctx.initialize();
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                    "No DirectML device available on this adapter");

            try (CpuReranker cpu = new CpuReranker(w, tok);
                 DirectMlReranker gpu = DirectMlReranker.build(ctx, /* ownsCtx */ false, w, tok)) {

                double sCpu = cpu.scorePair("query alpha", "document beta gamma");
                List<RerankResult> rGpu = gpu.rerank(new RerankRequest(
                        "query alpha", Arrays.asList("document beta gamma"), 1));
                double sGpu = rGpu.get(0).score();

                assertEquals(sCpu, sGpu, 1e-3,
                        "CPU↔DirectML reranker score must agree (cpu=" + sCpu + ", gpu=" + sGpu + ")");
            }
        }
    }

    @Test
    void cpuAndDirectMlPreserveDocumentRanking() throws Exception {
        assumeTrue(WindowsBindings.isSupported(),
                "DirectML requires Windows + D3D12 on this host");
        BertEncoderConfig cfg = tinyConfig();
        RerankerCpuWeights w = tinyWeights(cfg, 1337L);
        TinyPairTokenizer tok = new TinyPairTokenizer(cfg.vocabSize());

        try (DirectMlContextImpl ctx = new DirectMlContextImpl("reranker-parity-rank")) {
            ctx.initialize();
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                    "No DirectML device available on this adapter");

            try (CpuReranker cpu = new CpuReranker(w, tok);
                 DirectMlReranker gpu = DirectMlReranker.build(ctx, /* ownsCtx */ false, w, tok)) {

                List<String> docs = Arrays.asList(
                        "alpha bravo charlie",
                        "delta echo foxtrot",
                        "golf hotel india",
                        "juliet kilo lima");
                RerankRequest req = new RerankRequest("query xyz", docs, 0);
                List<RerankResult> cpuOrder = cpu.rerank(req);
                List<RerankResult> gpuOrder = gpu.rerank(req);
                assertEquals(cpuOrder.size(), gpuOrder.size());
                for (int i = 0; i < cpuOrder.size(); i++) {
                    assertEquals(cpuOrder.get(i).originalIndex(), gpuOrder.get(i).originalIndex(),
                            "ranking position " + i + " must match (cpu="
                                    + cpuOrder + ", gpu=" + gpuOrder + ")");
                    assertEquals(cpuOrder.get(i).score(), gpuOrder.get(i).score(), 1e-3,
                            "score at position " + i + " must match");
                }
                assertTrue(gpu.cachedStackCount() >= 1, "expected at least one cached stack");
            }
        }
    }

    private static float[] rand(Random r, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) (r.nextGaussian() * scale);
        return a;
    }

    private static float[] ones(int n) {
        float[] a = new float[n];
        Arrays.fill(a, 1f);
        return a;
    }

    private static float[] zeros(int n) {
        return new float[n];
    }
}

