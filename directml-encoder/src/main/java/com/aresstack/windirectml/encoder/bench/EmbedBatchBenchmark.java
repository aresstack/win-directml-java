package com.aresstack.windirectml.encoder.bench;

import com.aresstack.windirectml.encoder.EmbeddingException;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.EmbeddingRequest;
import com.aresstack.windirectml.encoder.EmbeddingVector;
import com.aresstack.windirectml.encoder.e5.E5Encoders;
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.encoder.minilm.CpuMiniLmEncoder;
import com.aresstack.windirectml.encoder.minilm.DirectMlMiniLmEncoder;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Throughput benchmark for the bucket-batched
 * {@link EmbeddingModel#embedBatch(List)} path.
 * <p>
 * For each {@code (backend, model)} combination it sweeps a configurable
 * set of batch sizes {@code N} and reports two numbers per row:
 * <ul>
 *   <li>{@code loopMs}  – total wall time of calling {@code embed(r)}
 *       {@code N} times sequentially.</li>
 *   <li>{@code batchMs} – total wall time of one
 *       {@code embedBatch(reqs)} call.</li>
 * </ul>
 * Per-text latency and the {@code batch/loop} speed-up are derived from
 * those two numbers. The benchmark is intentionally simple – no JMH, no
 * statistical smoothing – it only has to make the order-of-magnitude
 * difference between a sequential per-text path and the batched
 * {@code DirectMlBertEncoder.embedBatch} path visible.
 * <p>
 * Usage (all args optional):
 * <pre>
 *   gradle :directml-encoder:runEmbedBatchBenchmark \
 *          --args="model minilm both 1 10,50,100,500"
 *   gradle :directml-encoder:runEmbedBatchBenchmark \
 *          --args="model e5 dml 1 10,50,100"
 * </pre>
 * Positional args:
 * <ol>
 *   <li>{@code modelRoot}  – directory containing {@code all-MiniLM-L6-v2/}
 *       and/or an E5 model directory. Defaults to {@code "model"}.</li>
 *   <li>{@code which}      – {@code minilm}, {@code e5} or {@code both}.
 *       Defaults to {@code both}; missing model directories are skipped
 *       with a warning instead of failing the run.</li>
 *   <li>{@code backend}    – {@code cpu}, {@code dml} or {@code both}.
 *       Defaults to {@code both}; the DirectML pass auto-skips if the
 *       backend fails to build on this host.</li>
 *   <li>{@code warmup}     – number of warmup batches at {@code max(N)}.
 *       Defaults to {@code 1}.</li>
 *   <li>{@code sizes}      – CSV of {@code N} values. Defaults to
 *       {@code "10,50,100,500"}.</li>
 * </ol>
 */
public final class EmbedBatchBenchmark {

    private static final Set<String> ALLOWED_WHICH = Set.of("minilm", "e5", "both");
    private static final Set<String> ALLOWED_BACKEND = Set.of("cpu", "dml", "both");

    /**
     * Pairs a discovered E5 model directory with its matching {@link E5Variant}.
     */
    private record E5BenchmarkModel(Path dir, E5Variant variant) {
    }

    private EmbedBatchBenchmark() {
    }

    public static void main(String[] args) {
        Path modelRoot = args.length > 0 ? Path.of(args[0]) : Path.of("model");
        String which = args.length > 1 ? args[1].toLowerCase(Locale.ROOT) : "both";
        String backend = args.length > 2 ? args[2].toLowerCase(Locale.ROOT) : "both";
        int warmup = args.length > 3 ? Integer.parseInt(args[3]) : 1;
        int[] sizes = args.length > 4 ? parseCsv(args[4]) : new int[]{10, 50, 100, 500};

        if (!ALLOWED_WHICH.contains(which)) {
            throw new IllegalArgumentException(
                    "which must be one of " + ALLOWED_WHICH + ", was '" + which + "'");
        }
        if (!ALLOWED_BACKEND.contains(backend)) {
            throw new IllegalArgumentException(
                    "backend must be one of " + ALLOWED_BACKEND + ", was '" + backend + "'");
        }
        if (warmup < 0) {
            throw new IllegalArgumentException("warmup must be >= 0, was " + warmup);
        }
        if (sizes.length == 0) {
            throw new IllegalArgumentException("sizes must contain at least one entry");
        }
        for (int n : sizes) {
            if (n <= 0) {
                throw new IllegalArgumentException("every size must be > 0, was " + n
                        + " in " + csv(sizes));
            }
        }

        System.out.printf(Locale.ROOT,
                "EmbedBatchBenchmark: modelRoot=%s which=%s backend=%s warmup=%d sizes=%s%n",
                modelRoot, which, backend, warmup, csv(sizes));

        int maxN = Arrays.stream(sizes).max().orElseThrow();
        List<String> corpus = buildCorpus(maxN);

        boolean runMiniLm = which.equals("minilm") || which.equals("both");
        boolean runE5 = which.equals("e5") || which.equals("both");
        boolean runCpu = backend.equals("cpu") || backend.equals("both");
        boolean runDml = backend.equals("dml") || backend.equals("both");

        if (runMiniLm) {
            Path dir = modelRoot.resolve("all-MiniLM-L6-v2");
            if (!Files.isDirectory(dir)) {
                System.out.println("[skip] MiniLM model dir not found: " + dir);
            } else {
                if (runCpu) runCpuMiniLm(dir, corpus, sizes, warmup);
                if (runDml) runDmlMiniLm(dir, corpus, sizes, warmup);
            }
        }
        if (runE5) {
            E5BenchmarkModel e5 = resolveE5Model(modelRoot);
            if (e5 == null) {
                System.out.println("[skip] E5 model dir not found under " + modelRoot);
            } else {
                System.out.printf(Locale.ROOT, "E5 model: dir=%s variant=%s%n",
                        e5.dir(), e5.variant().token());
                if (runCpu) runCpuE5(e5, corpus, sizes, warmup);
                if (runDml) runDmlE5(e5, corpus, sizes, warmup);
            }
        }
    }

    // ── per (backend, model) runners ────────────────────────────────────

    private static void runCpuMiniLm(Path dir, List<String> corpus, int[] sizes, int warmup) {
        try (CpuMiniLmEncoder enc = CpuMiniLmEncoder.load(dir)) {
            runOne("cpu", "minilm", enc, /* prefix */ null, corpus, sizes, warmup);
        } catch (EmbeddingException e) {
            System.out.println("[skip] cpu/minilm: " + e.getMessage());
        }
    }

    private static void runDmlMiniLm(Path dir, List<String> corpus, int[] sizes, int warmup) {
        try (DirectMlMiniLmEncoder enc = DirectMlMiniLmEncoder.load(dir)) {
            runOne("dml", "minilm", enc, /* prefix */ null, corpus, sizes, warmup);
        } catch (Exception e) {
            System.out.println("[skip] dml/minilm: " + e.getMessage());
        }
    }

    private static void runCpuE5(E5BenchmarkModel model, List<String> corpus, int[] sizes, int warmup) {
        try (var enc = E5Encoders.loadCpu(model.dir(), model.variant())) {
            // E5 expects a task prefix; "passage:" matches the embed sidecar default.
            runOne("cpu", "e5", enc, "passage: ", corpus, sizes, warmup);
        } catch (EmbeddingException e) {
            System.out.println("[skip] cpu/e5: " + e.getMessage());
        }
    }

    private static void runDmlE5(E5BenchmarkModel model, List<String> corpus, int[] sizes, int warmup) {
        try (var enc = E5Encoders.loadDirectMl(model.dir(), model.variant())) {
            runOne("dml", "e5", enc, "passage: ", corpus, sizes, warmup);
        } catch (Exception e) {
            System.out.println("[skip] dml/e5: " + e.getMessage());
        }
    }

    // ── core measurement loop ───────────────────────────────────────────

    private static void runOne(String backend, String model,
                               EmbeddingModel enc, String prefix,
                               List<String> corpus, int[] sizes, int warmup) {
        System.out.printf(Locale.ROOT,
                "%n---- %s / %s ----%n", backend, model);

        // Warmup at max(N) so every pad-bucket cache entry is hot.
        int hot = Arrays.stream(sizes).max().orElseThrow();
        try {
            List<EmbeddingRequest> hotReqs = toRequests(corpus.subList(0, hot), prefix);
            for (int i = 0; i < warmup; i++) {
                enc.embedBatch(hotReqs);
            }
        } catch (EmbeddingException e) {
            System.out.println("[skip] " + backend + "/" + model + " warmup failed: " + e.getMessage());
            return;
        }

        System.out.printf(Locale.ROOT, "%-8s %-7s %6s %12s %12s %12s %12s %8s%n",
                "backend", "model", "N", "loopMs", "batchMs", "loopPerMs", "batchPerMs", "speedup");
        for (int n : sizes) {
            List<EmbeddingRequest> reqs = toRequests(corpus.subList(0, n), prefix);

            double loopMs;
            try {
                long t0 = System.nanoTime();
                for (EmbeddingRequest r : reqs) {
                    EmbeddingVector v = enc.embed(r);
                    Objects.requireNonNull(v); // prevent dead-code elimination
                }
                loopMs = (System.nanoTime() - t0) / 1_000_000.0;
            } catch (EmbeddingException e) {
                System.out.println("[skip] embed loop @N=" + n + ": " + e.getMessage());
                continue;
            }

            double batchMs;
            try {
                long t0 = System.nanoTime();
                List<EmbeddingVector> out = enc.embedBatch(reqs);
                Objects.requireNonNull(out);
                batchMs = (System.nanoTime() - t0) / 1_000_000.0;
            } catch (EmbeddingException e) {
                System.out.println("[skip] embedBatch @N=" + n + ": " + e.getMessage());
                continue;
            }

            double speedup = batchMs > 0 ? loopMs / batchMs : 0.0;
            System.out.printf(Locale.ROOT,
                    "%-8s %-7s %6d %12.2f %12.2f %12.3f %12.3f %7.2fx%n",
                    backend, model, n, loopMs, batchMs, loopMs / n, batchMs / n, speedup);
        }
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private static List<EmbeddingRequest> toRequests(List<String> texts, String prefix) {
        List<EmbeddingRequest> out = new ArrayList<>(texts.size());
        for (String t : texts) out.add(new EmbeddingRequest(t, /* normalize */ true, prefix));
        return out;
    }

    /**
     * Round-robin between four templates of distinct lengths so the
     * synthetic corpus realistically spans several pad buckets, mirroring
     * what a retrieval candidate set looks like in practice.
     */
    private static List<String> buildCorpus(int maxN) {
        String[] templates = {
                "DirectML accelerates ML on Windows GPUs.",
                "Paris is the capital of France.",
                "DirectML exposes Direct3D 12 device interfaces through IDMLDevice and "
                        + "IDMLOperator and is consumed by ONNX Runtime, PyTorch-DirectML, "
                        + "and TensorFlow-DirectML for cross-vendor GPU acceleration on "
                        + "Windows 11 systems – including discrete, integrated and Arm devices.",
                "Java 21 introduced virtual threads, pattern matching for switch, "
                        + "record patterns, sequenced collections and the Foreign Function "
                        + "and Memory API; together they meaningfully change how server "
                        + "applications interact with native libraries and concurrency."
        };
        List<String> out = new ArrayList<>(maxN);
        for (int i = 0; i < maxN; i++) {
            out.add(templates[i % templates.length] + " (doc #" + i + ")");
        }
        return out;
    }

    /**
     * Walk every supported {@link E5Variant} in priority order and return
     * the first variant whose directory hint resolves under
     * {@code modelRoot}. The benchmark only considers WordPiece-tokenizer
     * variants – SentencePiece-based {@code multilingual-e5-*} models are
     * deliberately skipped because the tokenizer is not yet supported.
     */
    private static E5BenchmarkModel resolveE5Model(Path modelRoot) {
        // Priority: prefer the small en/de model (project default), then
        // the english v2 line from small to large.
        E5Variant[] priority = {
                E5Variant.BASE_STS_EN_DE,
                E5Variant.SMALL_V2,
                E5Variant.BASE_V2,
                E5Variant.LARGE_V2
        };
        for (E5Variant v : priority) {
            for (Path hint : v.directoryHints()) {
                // Directory hints are "model/<name>"; strip the leading
                // "model/" so we can probe relative to the user-supplied
                // modelRoot (which usually IS "model" but may not be).
                Path tail = hint.getNameCount() > 1 ? hint.subpath(1, hint.getNameCount()) : hint;
                Path candidate = modelRoot.resolve(tail);
                if (Files.isDirectory(candidate)) {
                    return new E5BenchmarkModel(candidate, v);
                }
            }
        }
        return null;
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

