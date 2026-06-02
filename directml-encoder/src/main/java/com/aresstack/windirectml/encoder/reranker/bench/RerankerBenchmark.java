package com.aresstack.windirectml.encoder.reranker.bench;

import com.aresstack.windirectml.encoder.reranker.BertCrossEncoderRerankers;
import com.aresstack.windirectml.encoder.reranker.CpuReranker;
import com.aresstack.windirectml.encoder.reranker.DirectMlReranker;
import com.aresstack.windirectml.encoder.reranker.RerankRequest;
import com.aresstack.windirectml.encoder.reranker.Reranker;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Throughput benchmark for the cross-encoder reranker.
 * <p>
 * Sweeps {@code N ∈ {10, 50, 100}} on a synthetic mixed-length corpus
 * and reports per-pair latency for both the CPU and the DirectML
 * backend. Mainly intended as the evidence baseline for the
 * {@code perf(reranker)} sprint – before/after numbers can be diffed
 * by running it pre- and post-change.
 * <p>
 * Usage:
 * <pre>
 *   gradle :directml-encoder:runRerankerBenchmark \
 *          -PrerankerModelDir=model/cross-encoder-ms-marco-MiniLM-L-6-v2
 * </pre>
 * or directly via:
 * <pre>
 *   java ... RerankerBenchmark &lt;modelDir&gt; &lt;warmup&gt; &lt;sizes csv&gt;
 * </pre>
 * The benchmark is intentionally simple – no JMH, no statistical
 * smoothing – it only needs to make the order-of-magnitude difference
 * between sequential per-pair execution and bucket-batched DirectML reranking visible.
 */
public final class RerankerBenchmark {

    private static final String QUERY = "What is DirectML and how does it accelerate ML on Windows?";

    private RerankerBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Path modelDir = args.length > 0
                ? Path.of(args[0])
                : Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2");
        int warmup = args.length > 1 ? Integer.parseInt(args[1]) : 1;
        int[] sizes = args.length > 2 ? parseCsv(args[2]) : new int[]{10, 50, 100};

        List<String> corpus = buildCorpus(sizes[sizes.length - 1]);

        System.out.printf(Locale.ROOT,
                "RerankerBenchmark: modelDir=%s warmup=%d sizes=%s%n",
                modelDir, warmup, csv(sizes));

        try (CpuReranker cpu = BertCrossEncoderRerankers.loadCpu(modelDir)) {
            System.out.println("---- CPU backend ----");
            run("cpu", cpu, corpus, sizes, warmup);
        }
        try (DirectMlReranker dml = BertCrossEncoderRerankers.loadDirectMl(modelDir)) {
            System.out.println("---- DirectML backend ----");
            run("dml", dml, corpus, sizes, warmup);
        }
    }

    private static void run(String backend, Reranker reranker,
                            List<String> corpus, int[] sizes, int warmup) throws Exception {
        // One warmup pass over the largest N so every pad bucket is
        // pre-populated (otherwise the first call dominates the timing).
        int hot = sizes[sizes.length - 1];
        for (int i = 0; i < warmup; i++) {
            reranker.rerank(new RerankRequest(QUERY, corpus.subList(0, hot), hot));
        }
        System.out.printf(Locale.ROOT, "%-8s %8s %12s %12s%n",
                "backend", "N", "totalMs", "perPairMs");
        for (int n : sizes) {
            List<String> docs = corpus.subList(0, n);
            long t0 = System.nanoTime();
            reranker.rerank(new RerankRequest(QUERY, docs, n));
            long ns = System.nanoTime() - t0;
            double ms = ns / 1_000_000.0;
            System.out.printf(Locale.ROOT, "%-8s %8d %12.2f %12.3f%n",
                    backend, n, ms, ms / n);
        }
    }

    private static List<String> buildCorpus(int maxN) {
        // Round-robin between short-relevant, short-unrelated,
        // long-relevant, long-unrelated so a real call sees several
        // pad buckets just like a top-N retrieval candidate set would.
        String[] templates = {
                "DirectML accelerates ONNX models on Windows GPUs.",
                "Python is a general-purpose programming language.",
                "DirectML exposes Direct3D 12 device interfaces through IDMLDevice "
                        + "and IDMLOperator and is used by ONNX Runtime, PyTorch-DirectML "
                        + "and TensorFlow-DirectML for cross-vendor GPU acceleration on "
                        + "Windows 11 systems.",
                "Paris is the capital and most populous city of France, renowned "
                        + "for landmarks such as the Eiffel Tower and the Louvre, and "
                        + "for its long-standing culinary and cultural traditions."
        };
        List<String> out = new ArrayList<>(maxN);
        for (int i = 0; i < maxN; i++) {
            out.add(templates[i % templates.length] + " (doc #" + i + ")");
        }
        return out;
    }

    private static int[] parseCsv(String s) {
        String[] parts = s.split(",");
        int[] out = new int[parts.length];
        for (int i = 0; i < parts.length; i++) out[i] = Integer.parseInt(parts[i].trim());
        return out;
    }

    private static String csv(int[] xs) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < xs.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(xs[i]);
        }
        return sb.toString();
    }
}

