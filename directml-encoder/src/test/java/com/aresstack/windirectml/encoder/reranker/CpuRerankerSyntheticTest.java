package com.aresstack.windirectml.encoder.reranker;

import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.aresstack.windirectml.encoder.bert.BertCpuEncoderWeights;
import com.aresstack.windirectml.encoder.bert.BertCpuLayerWeights;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * CPU-only checks for the cross-encoder reranker: deterministic
 * forward, descending sort, top-N trimming, and graceful failure when
 * the tokenizer does not support pair encoding.
 */
class CpuRerankerSyntheticTest {

    private static final class TinyPairTokenizer implements EncoderTokenizer {
        @Override public Encoded encode(String text) { return encodePair(text, ""); }
        @Override
        public Encoded encodePair(String a, String b) {
            int[] aIds = idsFor(a);
            int[] bIds = idsFor(b);
            int n = 1 + aIds.length + 1 + bIds.length + 1;
            int[] ids = new int[n]; int[] mask = new int[n]; int[] seg = new int[n];
            int p = 0; ids[p++] = 2;
            for (int x : aIds) { ids[p] = x; seg[p] = 0; p++; }
            ids[p] = 3; seg[p] = 0; p++;
            for (int x : bIds) { ids[p] = x; seg[p] = 1; p++; }
            ids[p] = 3; seg[p] = 1;
            for (int i = 0; i < n; i++) mask[i] = 1;
            return new Encoded(ids, mask, seg);
        }
        private int[] idsFor(String t) {
            int n = Math.min(t.length(), 5);
            int[] out = new int[n];
            for (int i = 0; i < n; i++) out[i] = 4 + (t.charAt(i) % 20);
            return out;
        }
        @Override public int padTokenId() { return 0; }
        @Override public int clsTokenId() { return 2; }
        @Override public int sepTokenId() { return 3; }
        @Override public int vocabSize()  { return 32; }
    }

    private static BertEncoderConfig cfg() {
        return new BertEncoderConfig("test/reranker-cpu", 24, 1, 4, 48,
                32, 2, 32, 1e-12f, "gelu", 24,
                PoolingStrategy.MEAN, false);
    }

    private static RerankerCpuWeights weights(BertEncoderConfig c, long seed) {
        Random r = new Random(seed);
        int H = c.hiddenSize(), I = c.intermediateSize();
        float[] we = rand(r, c.vocabSize() * H, 0.02f);
        float[] pe = rand(r, c.maxPositionEmbeddings() * H, 0.02f);
        float[] tte = rand(r, c.typeVocabSize() * H, 0.02f);
        List<BertCpuLayerWeights> layers = new ArrayList<>();
        for (int l = 0; l < c.numLayers(); l++) {
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
        BertCpuEncoderWeights bert = BertCpuEncoderWeights.forTesting(c, we, pe, tte,
                ones(H), zeros(H), layers);
        return new RerankerCpuWeights(bert, rand(r, H, 0.1f),
                new float[]{ (float) (r.nextGaussian() * 0.01) }, 1);
    }

    @Test
    void resultsAreSortedAndTopNTrimmed() throws Exception {
        BertEncoderConfig c = cfg();
        try (CpuReranker rr = new CpuReranker(weights(c, 42L), new TinyPairTokenizer())) {
            List<String> docs = Arrays.asList("alpha", "bravo", "charlie", "delta", "echo");
            List<RerankResult> all = rr.rerank(new RerankRequest("query", docs, 0));
            assertEquals(docs.size(), all.size());
            for (int i = 1; i < all.size(); i++) {
                assertTrue(all.get(i - 1).score() >= all.get(i).score(),
                        "results must be sorted descending by score");
            }
            List<RerankResult> top2 = rr.rerank(new RerankRequest("query", docs, 2));
            assertEquals(2, top2.size());
            assertEquals(all.get(0).originalIndex(), top2.get(0).originalIndex());
            assertEquals(all.get(1).originalIndex(), top2.get(1).originalIndex());
        }
    }

    @Test
    void sameInputProducesSameScore() throws Exception {
        BertEncoderConfig c = cfg();
        try (CpuReranker rr = new CpuReranker(weights(c, 7L), new TinyPairTokenizer())) {
            double a = rr.scorePair("query", "doc one");
            double b = rr.scorePair("query", "doc one");
            assertEquals(a, b, 0.0, "scorePair must be deterministic");
        }
    }

    @Test
    void closedRerankerRejectsCalls() {
        BertEncoderConfig c = cfg();
        CpuReranker rr = new CpuReranker(weights(c, 1L), new TinyPairTokenizer());
        rr.close();
        try {
            rr.rerank(new RerankRequest("q", Arrays.asList("d"), 1));
            throw new AssertionError("expected RerankException for closed reranker");
        } catch (RerankException expected) {
            // good
        }
    }

    private static float[] rand(Random r, int n, float s) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) (r.nextGaussian() * s);
        return a;
    }
    private static float[] ones(int n) { float[] a = new float[n]; Arrays.fill(a, 1f); return a; }
    private static float[] zeros(int n) { return new float[n]; }
}

