package com.aresstack.windirectml.inference.bench;

import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import com.aresstack.windirectml.inference.Phi3InferenceEngine;

import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Minimaler Benchmark-Harness für Phi-3 (Issue 22).
 * <p>
 * Trennt Warmup von der Messphase und meldet:
 * <ul>
 *   <li>Modellladezeit</li>
 *   <li>Tokens pro Sekunde</li>
 *   <li>Latenz pro Generation</li>
 *   <li>Heap-Verbrauch (Java) und JVM-Argumente</li>
 * </ul>
 * Embedding-Durchsatz wird ergänzt, sobald die Encoder-Runtime aktiv ist.
 *
 * <p><b>Ausführung:</b>
 * <pre>
 *   java --enable-preview --enable-native-access=ALL-UNNAMED \
 *        -cp build/libs/* \
 *        com.aresstack.windirectml.inference.bench.Phi3Benchmark \
 *        [modelDir] [backend] [warmupRuns] [measureRuns]
 * </pre>
 *
 * <p>Ausgabe ist auf stderr, damit die Klasse problemlos auch aus
 * einem Skript heraus aufgerufen werden kann.
 */
public final class Phi3Benchmark {

    private final Path modelDir;
    private final String backend;
    private final int warmupRuns;
    private final int measureRuns;
    private final int maxTokens;

    public Phi3Benchmark(Path modelDir, String backend, int warmupRuns, int measureRuns, int maxTokens) {
        this.modelDir = modelDir;
        this.backend = backend;
        this.warmupRuns = warmupRuns;
        this.measureRuns = measureRuns;
        this.maxTokens = maxTokens;
    }

    public Map<String, Object> run() throws InferenceException {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("hardware", hardwareInfo());
        report.put("jvmArgs", ManagementFactory.getRuntimeMXBean().getInputArguments());
        report.put("modelDir", modelDir.toString());
        report.put("backend", backend);
        report.put("maxTokens", maxTokens);

        Phi3InferenceEngine engine = new Phi3InferenceEngine(modelDir, maxTokens, backend);

        long loadStart = System.nanoTime();
        engine.initialize();
        long loadElapsedMs = (System.nanoTime() - loadStart) / 1_000_000;
        report.put("modelLoadMs", loadElapsedMs);

        try {
            // Warmup
            for (int i = 0; i < warmupRuns; i++) {
                engine.generate(prompt("warmup " + i));
            }

            // Measure
            long totalNs = 0;
            int totalTokens = 0;
            for (int i = 0; i < measureRuns; i++) {
                long t0 = System.nanoTime();
                InferenceResult res = engine.generate(prompt("measure " + i));
                totalNs += System.nanoTime() - t0;
                if (res.getUsage() != null) totalTokens += res.getUsage().completionTokens();
            }
            double totalMs = totalNs / 1_000_000.0;
            double tokensPerSec = totalMs > 0 ? totalTokens / (totalMs / 1000.0) : 0;

            report.put("measureRuns", measureRuns);
            report.put("warmupRuns", warmupRuns);
            report.put("totalTokens", totalTokens);
            report.put("totalElapsedMs", round(totalMs));
            report.put("avgLatencyMs", round(totalMs / Math.max(measureRuns, 1)));
            report.put("tokensPerSec", round(tokensPerSec));
            return report;
        } finally {
            engine.shutdown();
        }
    }

    private static InferenceRequest prompt(String label) {
        return InferenceRequest.builder()
                .systemPrompt("You are a benchmark assistant. Respond very briefly.")
                .userPrompt("Repeat the word 'ok' five times. Tag: " + label)
                .maxTokens(32)
                .build();
    }

    private static Map<String, Object> hardwareInfo() {
        Map<String, Object> hw = new LinkedHashMap<>();
        hw.put("os", System.getProperty("os.name") + " " + System.getProperty("os.version"));
        hw.put("arch", System.getProperty("os.arch"));
        hw.put("availableProcessors", Runtime.getRuntime().availableProcessors());
        hw.put("maxHeapMb", Runtime.getRuntime().maxMemory() / (1024 * 1024));
        hw.put("javaVendor", System.getProperty("java.vendor"));
        hw.put("javaVersion", System.getProperty("java.version"));
        return hw;
    }

    private static double round(double v) {
        return Math.round(v * 100.0) / 100.0;
    }

    public static void main(String[] args) throws Exception {
        Path modelDir = args.length > 0 ? Path.of(args[0])
                : Path.of("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        String backend = args.length > 1 ? args[1] : "auto";
        int warmup = args.length > 2 ? Integer.parseInt(args[2]) : 1;
        int measure = args.length > 3 ? Integer.parseInt(args[3]) : 5;

        Phi3Benchmark bench = new Phi3Benchmark(modelDir, backend, warmup, measure, 32);
        Map<String, Object> report = bench.run();
        report.forEach((k, v) -> System.err.println(k + ": " + v));
    }
}

