package com.aresstack.windirectml.encoder.e5;

import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.EncoderTokenizer;
import com.aresstack.windirectml.encoder.PoolingStrategy;
import com.aresstack.windirectml.encoder.DirectMlTestAssumptions;
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
 * Parity test for the bucket-batched {@code DirectMlBertEncoder.embedBatch}
 * code path. Uses a synthetic tiny E5-shaped config (same fixture as
 * {@code E5SyntheticParityTest}) and asserts:
 *
 * <ol>
 *   <li>{@code embedBatch} returns one vector per input in input order.</li>
 *   <li>Each batched DirectML vector matches the CPU per-text {@code embed}
 *       output by cosine &gt; 0.999.</li>
 *   <li>Each batched DirectML vector matches the per-call DirectML
 *       {@code embed} output (same backend, batch=1 path) by cosine
 *       &gt; 0.9999 – proves the batched path computes the same graph,
 *       not just a similar one.</li>
 *   <li>The batched stack cache reflects {@code (bucket, batch)} keying:
 *       multiple inputs in one bucket ⇒ exactly one batched stack.</li>
 * </ol>
 */
@EnabledOnOs(OS.WINDOWS)
class DirectMlEmbedBatchParityTest {

    /**
     * Same TinyTokenizer fixture as the existing synthetic parity test.
     */
    private static final class TinyTokenizer implements EncoderTokenizer {
        private final int vocab;

        TinyTokenizer(int vocab) {
            this.vocab = vocab;
        }

        @Override
        public Encoded encode(String text) {
            int n = Math.min(text.length() + 2, 8);
            int[] ids = new int[n];
            int[] mask = new int[n];
            int[] segs = new int[n];
            ids[0] = 2;
            for (int i = 1; i < n - 1; i++) ids[i] = 4 + (text.charAt(i - 1) % (vocab - 5));
            ids[n - 1] = 3;
            for (int i = 0; i < n; i++) mask[i] = 1;
            return new Encoded(ids, mask, segs);
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
                "test/e5-tiny",
                24, 2, 4, 48,
                16, 2, 32,
                1e-12f, "gelu", 24,
                PoolingStrategy.MEAN, true);
    }

    private static BertCpuEncoderWeights randomWeights(BertEncoderConfig cfg, long seed) {
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
        return BertCpuEncoderWeights.forTesting(cfg, we, pe, tte, elng, elnb, layers);
    }

    @Test
    void batchedEmbedMatchesSequentialAcrossBackends() throws Exception {
        assumeTrue(WindowsBindings.isSupported(),
                "DirectML requires Windows + D3D12 on this host");
        BertEncoderConfig cfg = tinyConfig();
        BertCpuEncoderWeights w = randomWeights(cfg, 9001L);
        TinyTokenizer tok = new TinyTokenizer(cfg.vocabSize());

        // All inputs have token-length <= 8 → all fall into the smallest
        // pad-bucket (S=64). One batched stack must materialise.
        List<String> texts = List.of(
                "a", "ab", "abc", "abcd", "abcde", "abcdef");
        List<EmbeddingRequest> reqs = new ArrayList<>();
        for (String t : texts) reqs.add(new EmbeddingRequest(t, true, null));

        try {
            try (DirectMlContextImpl ctx = new DirectMlContextImpl("embedBatch-parity")) {
                ctx.initialize();
                assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                        "No DirectML device available on this adapter");

                try (CpuBertEncoder cpu = new CpuBertEncoder(cfg, w, tok);
                     DirectMlBertEncoder gpu = DirectMlBertEncoder.build(
                             ctx, /* ownsCtx */ false, cfg, w, tok)) {

                // 1) Reference: CPU sequential.
                List<EmbeddingVector> cpuSeq = new ArrayList<>();
                for (EmbeddingRequest r : reqs) cpuSeq.add(cpu.embed(r));

                // 2) DirectML sequential (existing path, batch=1).
                List<EmbeddingVector> gpuSeq = new ArrayList<>();
                for (EmbeddingRequest r : reqs) gpuSeq.add(gpu.embed(r));

                int batchCacheBefore = gpu.cachedBatchStackCount();

                // 3) DirectML batched (the new path).
                List<EmbeddingVector> gpuBatch = gpu.embedBatch(reqs);

                // (a) Order + count preserved.
                assertEquals(reqs.size(), gpuBatch.size(),
                        "embedBatch must return one vector per input");
                for (int i = 0; i < reqs.size(); i++) {
                    assertEquals(cfg.outputDimension(), gpuBatch.get(i).dimension());
                    assertTrue(Math.abs(norm(gpuBatch.get(i).values()) - 1.0) < 1e-4,
                            "embedBatch output #" + i + " must be unit-norm");
                }

                // (b) Per-row parity vs CPU sequential.
                for (int i = 0; i < reqs.size(); i++) {
                    double cos = CosineSimilarity.compute(
                            cpuSeq.get(i).values(), gpuBatch.get(i).values());
                    assertTrue(cos > 0.999,
                            "CPU-seq vs DML-batch cosine[" + i + "] must be > 0.999, was " + cos);
                }

                // (c) Per-row parity vs DirectML sequential (same backend).
                for (int i = 0; i < reqs.size(); i++) {
                    double cos = CosineSimilarity.compute(
                            gpuSeq.get(i).values(), gpuBatch.get(i).values());
                    assertTrue(cos > 0.9999,
                            "DML-seq vs DML-batch cosine[" + i + "] must be > 0.9999, was " + cos);
                }

                // (d) Exactly one new batched stack was built (single bucket, single N).
                int batchCacheAfter = gpu.cachedBatchStackCount();
                assertEquals(batchCacheBefore + 1, batchCacheAfter,
                        "embedBatch must materialise exactly one (bucket, batch) entry "
                                + "for a single-bucket / single-N call");
                }
            }
        } catch (Exception e) {
            DirectMlTestAssumptions.skipIfHostDirectMlUnavailable(e);
            throw e;
        }
    }

    @Test
    void batchedEmbedUsesDistinctCacheEntriesForDifferentBatchSizes() throws Exception {
        assumeTrue(WindowsBindings.isSupported(),
                "DirectML requires Windows + D3D12 on this host");
        BertEncoderConfig cfg = tinyConfig();
        BertCpuEncoderWeights w = randomWeights(cfg, 12345L);
        TinyTokenizer tok = new TinyTokenizer(cfg.vocabSize());

        // The TinyTokenizer caps length at 8 → all inputs land in the same
        // bucket. To prove the (bucket, batch) cache really keys on both
        // dimensions we feed the encoder twice with different N values and
        // assert two distinct cache entries appear.
        try {
            try (DirectMlContextImpl ctx = new DirectMlContextImpl("embedBatch-batch-sizes")) {
                ctx.initialize();
                assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                        "No DirectML device available on this adapter");

                try (DirectMlBertEncoder gpu = DirectMlBertEncoder.build(
                        ctx, /* ownsCtx */ false, cfg, w, tok)) {
                List<EmbeddingRequest> two = List.of(
                        new EmbeddingRequest("alpha", true, null),
                        new EmbeddingRequest("beta", true, null));
                List<EmbeddingRequest> three = List.of(
                        new EmbeddingRequest("gamma", true, null),
                        new EmbeddingRequest("delta", true, null),
                        new EmbeddingRequest("eps", true, null));

                List<EmbeddingVector> r2 = gpu.embedBatch(two);
                List<EmbeddingVector> r3 = gpu.embedBatch(three);

                assertEquals(2, r2.size());
                assertEquals(3, r3.size());
                assertEquals(2, gpu.cachedBatchStackCount(),
                        "two distinct (bucket=B, batch=2) and (bucket=B, batch=3) "
                                + "entries expected, even though the bucket is identical");
                }
            }
        } catch (Exception e) {
            DirectMlTestAssumptions.skipIfHostDirectMlUnavailable(e);
            throw e;
        }
    }

    /**
     * Variable-length tokenizer used by the real multi-bucket test below.
     */
    private static final class LengthTokenizer implements EncoderTokenizer {
        private final int vocab;
        private final int maxLen;

        LengthTokenizer(int vocab, int maxLen) {
            this.vocab = vocab;
            this.maxLen = maxLen;
        }

        @Override
        public Encoded encode(String text) {
            int n = Math.min(text.length() + 2, maxLen);
            int[] ids = new int[n];
            int[] mask = new int[n];
            int[] segs = new int[n];
            ids[0] = 2;
            for (int i = 1; i < n - 1; i++) ids[i] = 4 + ((text.charAt((i - 1) % text.length())) % (vocab - 5));
            ids[n - 1] = 3;
            for (int i = 0; i < n; i++) mask[i] = 1;
            return new Encoded(ids, mask, segs);
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

    /**
     * Larger config that allows seqLen > 16 so we can land in multiple pad buckets.
     */
    private static BertEncoderConfig multiBucketConfig() {
        return new BertEncoderConfig(
                "test/e5-multibucket",
                24, 2, 4, 48,
                /* maxPos */ 128, 2, 32,
                1e-12f, "gelu", 24,
                PoolingStrategy.MEAN, true);
    }

    private static String repeat(String s, int n) {
        StringBuilder sb = new StringBuilder(s.length() * n);
        for (int i = 0; i < n; i++) sb.append(s);
        return sb.toString();
    }

    @Test
    void batchedEmbedSpansMultipleBuckets() throws Exception {
        assumeTrue(WindowsBindings.isSupported(),
                "DirectML requires Windows + D3D12 on this host");
        BertEncoderConfig cfg = multiBucketConfig();
        BertCpuEncoderWeights w = randomWeights(cfg, 77L);
        LengthTokenizer tok = new LengthTokenizer(cfg.vocabSize(), cfg.maxPositionEmbeddings());

        // Inputs deliberately straddle two pad buckets:
        //   * short text  → seqLen ≈ 5  → bucket 64
        //   * medium text → seqLen ≈ 100 → bucket 128
        String shortText = "abc";                                   // length 3 → 5 tokens
        String mediumText = repeat("x", 100);                         // length 100 → 102 tokens
        List<EmbeddingRequest> mixed = List.of(
                new EmbeddingRequest(shortText, true, null),
                new EmbeddingRequest(mediumText, true, null),
                new EmbeddingRequest(shortText + "!", true, null),   // also bucket 64
                new EmbeddingRequest(mediumText.substring(0, 80), true, null) // bucket 128
        );

        try (DirectMlContextImpl ctx = new DirectMlContextImpl("embedBatch-multi-bucket")) {
            ctx.initialize();
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                    "No DirectML device available on this adapter");

            try (CpuBertEncoder cpu = new CpuBertEncoder(cfg, w, tok);
                 DirectMlBertEncoder gpu = DirectMlBertEncoder.build(
                         ctx, /* ownsCtx */ false, cfg, w, tok)) {

                // Reference: CPU sequential. Establishes per-row truth.
                List<EmbeddingVector> cpuSeq = new ArrayList<>();
                for (EmbeddingRequest r : mixed) cpuSeq.add(cpu.embed(r));

                // The batched call must dispatch once per active bucket.
                int before = gpu.cachedBatchStackCount();
                List<EmbeddingVector> gpuBatch = gpu.embedBatch(mixed);
                int after = gpu.cachedBatchStackCount();

                // Two short + two medium inputs → exactly two (bucket, N=2) entries.
                assertEquals(before + 2, after,
                        "two pad buckets in the input must materialise two batched stacks");

                // Order preserved + parity with CPU.
                assertEquals(mixed.size(), gpuBatch.size());
                for (int i = 0; i < mixed.size(); i++) {
                    assertEquals(cfg.outputDimension(), gpuBatch.get(i).dimension());
                    double cos = CosineSimilarity.compute(
                            cpuSeq.get(i).values(), gpuBatch.get(i).values());
                    assertTrue(cos > 0.999,
                            "multi-bucket batched cos[" + i + "] vs CPU must be > 0.999, was " + cos);
                }
            }
        } catch (Exception e) {
            DirectMlTestAssumptions.skipIfHostDirectMlUnavailable(e);
            throw e;
        }
    }

    /**
     * {@code embedBatch} with {@code normalize=false} for every row must
     * skip the GPU L2 dispatch and return un-normalised pooled vectors
     * that still match the CPU reference in direction (cos &gt; 0.999).
     */
    @Test
    void batchedEmbedNormalizeFalseSkipsGpuL2() throws Exception {
        assumeTrue(WindowsBindings.isSupported(),
                "DirectML requires Windows + D3D12 on this host");
        BertEncoderConfig cfg = tinyConfig();
        BertCpuEncoderWeights w = randomWeights(cfg, 4242L);
        TinyTokenizer tok = new TinyTokenizer(cfg.vocabSize());

        List<EmbeddingRequest> reqs = new ArrayList<>();
        for (String t : List.of("a", "ab", "abc", "abcd")) {
            reqs.add(new EmbeddingRequest(t, /* normalize */ false, null));
        }

        try (DirectMlContextImpl ctx = new DirectMlContextImpl("embedBatch-normalize-false")) {
            ctx.initialize();
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                    "No DirectML device available on this adapter");

            try (CpuBertEncoder cpu = new CpuBertEncoder(cfg, w, tok);
                 DirectMlBertEncoder gpu = DirectMlBertEncoder.build(
                         ctx, /* ownsCtx */ false, cfg, w, tok)) {

                List<EmbeddingVector> cpuSeq = new ArrayList<>();
                for (EmbeddingRequest r : reqs) cpuSeq.add(cpu.embed(r));

                List<EmbeddingVector> gpuBatch = gpu.embedBatch(reqs);

                assertEquals(reqs.size(), gpuBatch.size(),
                        "embedBatch must return one vector per input");
                for (int i = 0; i < reqs.size(); i++) {
                    EmbeddingVector v = gpuBatch.get(i);
                    assertEquals(cfg.outputDimension(), v.dimension());
                    assertEquals(false, v.normalized(),
                            "normalize=false row #" + i + " must report normalized() == false");
                    double nrm = norm(v.values());
                    assertTrue(Math.abs(nrm - 1.0) > 1e-3,
                            "normalize=false row #" + i + " must NOT be unit-norm, was " + nrm);
                    double cos = CosineSimilarity.compute(
                            cpuSeq.get(i).values(), v.values());
                    assertTrue(cos > 0.999,
                            "CPU vs DML-batch cosine[" + i + "] (normalize=false) must be > 0.999, was " + cos);
                }
            }
        } catch (Exception e) {
            DirectMlTestAssumptions.skipIfHostDirectMlUnavailable(e);
            throw e;
        }
    }

    /**
     * Mixed {@code normalize} flags within one batch: rows flagged
     * {@code true} must be unit-norm and rows flagged {@code false} must
     * not be. Both groups must respect the original input order.
     */
    @Test
    void batchedEmbedSupportsMixedNormalizeFlags() throws Exception {
        assumeTrue(WindowsBindings.isSupported(),
                "DirectML requires Windows + D3D12 on this host");
        BertEncoderConfig cfg = tinyConfig();
        BertCpuEncoderWeights w = randomWeights(cfg, 9999L);
        TinyTokenizer tok = new TinyTokenizer(cfg.vocabSize());

        // All inputs share the same pad bucket on purpose, so the mixed
        // flags exercise the (anyNormalize && !allNormalize) double-readback
        // branch in embedBatch().
        List<EmbeddingRequest> reqs = List.of(
                new EmbeddingRequest("alpha",   /* normalize */ true, null),
                new EmbeddingRequest("beta",    /* normalize */ false, null),
                new EmbeddingRequest("gamma",   /* normalize */ true, null),
                new EmbeddingRequest("delta",   /* normalize */ false, null));

        try (DirectMlContextImpl ctx = new DirectMlContextImpl("embedBatch-mixed-normalize")) {
            ctx.initialize();
            assumeTrue(ctx.isReady() && ctx.bindings().hasDirectMl(),
                    "No DirectML device available on this adapter");

            try (CpuBertEncoder cpu = new CpuBertEncoder(cfg, w, tok);
                 DirectMlBertEncoder gpu = DirectMlBertEncoder.build(
                         ctx, /* ownsCtx */ false, cfg, w, tok)) {

                // CPU reference per row, respecting each row's normalize flag.
                List<EmbeddingVector> cpuSeq = new ArrayList<>();
                for (EmbeddingRequest r : reqs) cpuSeq.add(cpu.embed(r));

                List<EmbeddingVector> gpuBatch = gpu.embedBatch(reqs);

                assertEquals(reqs.size(), gpuBatch.size(),
                        "embedBatch must return one vector per input");
                for (int i = 0; i < reqs.size(); i++) {
                    boolean expectNormalized = reqs.get(i).normalize();
                    EmbeddingVector v = gpuBatch.get(i);
                    assertEquals(expectNormalized, v.normalized(),
                            "row #" + i + " normalized() flag must match request");
                    double nrm = norm(v.values());
                    if (expectNormalized) {
                        assertTrue(Math.abs(nrm - 1.0) < 1e-4,
                                "normalize=true row #" + i + " must be unit-norm, was " + nrm);
                    } else {
                        assertTrue(Math.abs(nrm - 1.0) > 1e-3,
                                "normalize=false row #" + i + " must NOT be unit-norm, was " + nrm);
                    }
                    double cos = CosineSimilarity.compute(
                            cpuSeq.get(i).values(), v.values());
                    assertTrue(cos > 0.999,
                            "mixed-normalize batched cos[" + i + "] vs CPU must be > 0.999, was " + cos);
                }
            }
        } catch (Exception e) {
            DirectMlTestAssumptions.skipIfHostDirectMlUnavailable(e);
            throw e;
        }
    }

    private static float[] rand(Random r, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) (r.nextGaussian() * scale);
        return a;
    }

    private static float[] ones(int n) {
        float[] a = new float[n];
        java.util.Arrays.fill(a, 1f);
        return a;
    }

    private static float[] zeros(int n) {
        return new float[n];
    }

    private static double norm(float[] v) {
        double s = 0;
        for (float x : v) s += (double) x * x;
        return Math.sqrt(s);
    }
}
