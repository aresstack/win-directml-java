package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.config.models.EmbeddingModelRegistry;
import com.aresstack.windirectml.encoder.EmbeddingModel;
import com.aresstack.windirectml.encoder.bert.BertEncoderConfig;
import com.aresstack.windirectml.encoder.e5.E5Encoders;
import com.aresstack.windirectml.encoder.e5.E5Variant;
import com.aresstack.windirectml.encoder.minilm.CpuMiniLmEncoder;
import com.aresstack.windirectml.encoder.minilm.DirectMlMiniLmEncoder;
import com.aresstack.windirectml.encoder.reranker.BertCrossEncoderRerankers;
import com.aresstack.windirectml.encoder.reranker.Reranker;
import com.aresstack.windirectml.inference.Phi3InferenceEngine;
import com.aresstack.windirectml.inference.Phi3Summarizer;
import com.aresstack.windirectml.inference.Summarizer;
import com.aresstack.windirectml.sidecar.handlers.CancelHandler;
import com.aresstack.windirectml.sidecar.handlers.EmbedHandler;
import com.aresstack.windirectml.sidecar.handlers.EmbedBatchHandler;
import com.aresstack.windirectml.sidecar.handlers.HealthHandler;
import com.aresstack.windirectml.sidecar.handlers.RerankHandler;
import com.aresstack.windirectml.sidecar.handlers.ShutdownHandler;
import com.aresstack.windirectml.sidecar.handlers.SummarizeHandler;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcError;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcErrorCode;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMessageReader;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMessageReader.RawLine;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMessageWriter;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcNotification;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcRequest;
import com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcResponse;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.NullNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Java-21 DirectML Phi-3 sidecar – JSON-RPC 2.0 over stdin/stdout.
 * <p>
 * <b>Transport contract</b>
 * <ul>
 *   <li>stdin: one JSON-RPC request per line.</li>
 *   <li>stdout: one JSON-RPC response or notification per line.
 *       <b>Nothing else</b> is written to stdout, ever.</li>
 *   <li>stderr: logs, stack traces, diagnostics.</li>
 * </ul>
 * <b>Initial methods:</b> {@code health}, {@code summarize}, {@code embed},
 * {@code shutdown}, {@code cancel}.
 */
public final class DirectMlPhi3Sidecar {

    private static final Logger log = LoggerFactory.getLogger(DirectMlPhi3Sidecar.class);

    private final InputStream in;
    private final OutputStream out;
    private final Path modelDir;
    private final String backend;
    private final int defaultMaxTokens;
    private final boolean autoLoadModel;

    private final ObjectMapper mapper = new ObjectMapper();
    private final SidecarStatus status = new SidecarStatus();
    private final SidecarCommandDispatcher dispatcher = new SidecarCommandDispatcher();
    private final AtomicReference<EmbeddingModel> embeddingModel = new AtomicReference<>();
    private final AtomicReference<Reranker> reranker = new AtomicReference<>();

    private Phi3Summarizer ownedSummarizer;
    private Summarizer summarizer;

    /**
     * Production constructor: own and load the Phi-3 summarizer on a background thread.
     */
    public DirectMlPhi3Sidecar(InputStream in, OutputStream out, Path modelDir,
                               String backend, int defaultMaxTokens) {
        this(in, out, modelDir, backend, defaultMaxTokens, true);
    }

    /**
     * Test constructor: skip model load, use stub or pre-built {@link Summarizer}.
     */
    public DirectMlPhi3Sidecar(InputStream in, OutputStream out, Path modelDir,
                               String backend, int defaultMaxTokens, boolean autoLoadModel) {
        this.in = in;
        this.out = out;
        this.modelDir = modelDir;
        this.backend = backend != null ? backend : "auto";
        this.defaultMaxTokens = defaultMaxTokens > 0 ? defaultMaxTokens : 512;
        this.autoLoadModel = autoLoadModel;
    }

    /**
     * Inject a pre-built summarizer instead of constructing one. Useful for tests.
     */
    public DirectMlPhi3Sidecar withSummarizer(Summarizer external) {
        this.summarizer = external;
        if (external != null && external.isReady()) {
            status.setModelLoaded(true);
            status.setMode("injected");
            status.setSummarizerReady(true);
            status.setSummarizerBackend("injected");
        }
        return this;
    }

    /**
     * Register an {@link EmbeddingModel} so that {@code embed} returns real vectors.
     * The backend name is reported as {@code "custom"} in {@code health}.
     */
    public DirectMlPhi3Sidecar withEmbeddingModel(EmbeddingModel model) {
        return withEmbeddingBackend("custom", model);
    }

    /**
     * Register an {@link EmbeddingModel} with an explicit backend name
     * ({@code "cpu"}, {@code "directml"}, …) used in {@code health}.
     */
    public DirectMlPhi3Sidecar withEmbeddingBackend(String backendName, EmbeddingModel model) {
        this.embeddingModel.set(model);
        status.setEmbeddingBackend(model != null ? backendName : null);
        status.setEmbeddingReady(model != null && model.isReady());
        return this;
    }

    /**
     * Register a {@link Reranker} so that {@code rerank} returns real
     * scores. The backend name is reported in {@code health} as
     * {@code rerankerBackend}.
     */
    public DirectMlPhi3Sidecar withRerankerBackend(String backendName, Reranker model) {
        this.reranker.set(model);
        status.setRerankerBackend(model != null ? backendName : null);
        status.setRerankerReady(model != null && model.isReady());
        status.setRerankerModel(model != null ? model.modelName() : null);
        return this;
    }

    /**
     * Test-only accessor for the mutable status object.
     */
    SidecarStatus statusForTesting() {
        return status;
    }

    // ── Entry point ──────────────────────────────────────────────────────

    public static void main(String[] args) {
        Path modelDir = resolveModelDir();
        String backend = System.getProperty("phi3.backend", "auto");
        int maxTokens = Integer.getInteger("phi3.maxTokens", 512);

        DirectMlPhi3Sidecar sidecar = new DirectMlPhi3Sidecar(
                System.in, System.out, modelDir, backend, maxTokens, true);

        // Embedding backend: parse -Dembed.backend first so that a forced
        // mode (cpu/directml) fails visibly even when the model directory
        // is missing entirely. -Dembed.model selects the encoder family
        // (default: minilm; supported: minilm, e5).
        EmbeddingBackendSelector.Mode embedMode;
        try {
            embedMode = EmbeddingBackendSelector.Mode.parse(System.getProperty("embed.backend"));
        } catch (IllegalArgumentException e) {
            log.error("Invalid -Dembed.backend value: {}", e.getMessage());
            System.exit(2);
            return;
        }
        String embedFamily;
        try {
            embedFamily = embedFamily(System.getProperty("embed.model"));
        } catch (IllegalArgumentException e) {
            log.error("Invalid -Dembed.model value: {}", e.getMessage());
            System.exit(2);
            return;
        }

        Path embedModelDir;
        EmbeddingBackendSelector.EncoderLoader cpuLoader;
        EmbeddingBackendSelector.EncoderLoader dmlLoader;
        String missingMsg;
        switch (embedFamily) {
            case "e5": {
                E5Variant e5Variant;
                try {
                    e5Variant = E5Variant.parse(System.getProperty("e5.model"));
                } catch (IllegalArgumentException e) {
                    log.error("Invalid -De5.model value: {}", e.getMessage());
                    System.exit(2);
                    return;
                }
                embedModelDir = resolveE5Dir(e5Variant);
                final E5Variant fv = e5Variant;
                cpuLoader = dir -> E5Encoders.loadCpu(dir, fv);
                dmlLoader = dir -> E5Encoders.loadDirectMl(dir, fv);
                missingMsg = "No E5 model directory found for variant=" + e5Variant.token()
                        + " (checked -De5.modelDir and " + e5Variant.directoryHints() + ")";
                log.info("Embedding family=e5 variant={} (-De5.model)", e5Variant.token());
                break;
            }
            case "minilm":
            default: {
                embedFamily = "minilm";
                embedModelDir = resolveMiniLmDir();
                cpuLoader = CpuMiniLmEncoder::load;
                dmlLoader = DirectMlMiniLmEncoder::load;
                missingMsg = "No MiniLM model directory found "
                        + "(checked -Dminilm.modelDir and model/all-MiniLM-L6-v2/)";
                break;
            }
        }

        if (embedModelDir == null) {
            if (embedMode == EmbeddingBackendSelector.Mode.CPU
                    || embedMode == EmbeddingBackendSelector.Mode.DIRECTML) {
                log.error("embed.backend={} (model={}) requested but {}",
                        embedMode.token(), embedFamily, missingMsg);
                sidecar.status.setEmbeddingBackend("error");
                sidecar.status.setEmbeddingReady(false);
                sidecar.status.setLastError("embed.backend=" + embedMode.token()
                        + " (model=" + embedFamily + ") requested but "
                        + missingMsg.toLowerCase(java.util.Locale.ROOT));
                System.exit(3);
                return;
            }
            log.info("{} – embed handler will report not-implemented", missingMsg);
        } else {
            log.info("{} model present at {} – selecting embedding backend (mode={})",
                    embedFamily, embedModelDir, embedMode.token());
            EmbeddingBackendSelector selector = new EmbeddingBackendSelector(cpuLoader, dmlLoader);
            try {
                EmbeddingBackendSelector.Selection sel = selector.select(embedMode, embedModelDir);
                sidecar.withEmbeddingBackend(sel.backend(), sel.model());
                if (sel.fallback()) {
                    // Surface the auto-fallback as a dedicated, non-error
                    // signal in health so the workbench/client can show
                    // *why* we ended up on CPU even though the user did
                    // not force it. lastError stays reserved for hard
                    // failures only.
                    sidecar.status.setEmbeddingFallback(true);
                    sidecar.status.setEmbeddingFallbackReason(sel.warning());
                }
                log.info("Embedding backend ready: family={} backend={} (fallback={})",
                        embedFamily, sel.backend(), sel.fallback());
            } catch (RuntimeException e) {
                log.error("Embedding backend initialisation failed: {}", e.getMessage(), e);
                sidecar.status.setEmbeddingBackend("error");
                sidecar.status.setEmbeddingReady(false);
                sidecar.status.setLastError("embed.backend=" + embedMode.token()
                        + " (model=" + embedFamily + ") failed: " + e.getMessage());
                if (embedMode == EmbeddingBackendSelector.Mode.DIRECTML
                        || embedMode == EmbeddingBackendSelector.Mode.CPU) {
                    System.exit(3);
                    return;
                }
            }
        }

        // Reranker (optional). When no model directory is present and
        // backend=auto, the rerank handler stays in NOT_IMPLEMENTED mode –
        // same fallback pattern as embed. Forced modes (cpu/directml) fail
        // visibly (exit 3) when the model is missing or the backend cannot
        // be loaded; an unknown -Drerank.backend value aborts with exit 2.
        RerankerBackendMode rerankMode;
        try {
            rerankMode = RerankerBackendMode.parse(System.getProperty("rerank.backend"));
        } catch (IllegalArgumentException e) {
            log.error("Invalid -Drerank.backend value: {}", e.getMessage());
            System.exit(2);
            return;
        }
        Path rerankDir = resolveRerankerDir();
        String rerankMissingMsg = "No reranker model directory found "
                + "(checked -Drerank.modelDir and "
                + "model/cross-encoder-ms-marco-MiniLM-L-6-v2/)";
        if (rerankDir == null) {
            if (rerankMode == RerankerBackendMode.CPU
                    || rerankMode == RerankerBackendMode.DIRECTML) {
                log.error("rerank.backend={} requested but {}",
                        rerankMode.token(), rerankMissingMsg);
                sidecar.status.setRerankerBackend("error");
                sidecar.status.setRerankerReady(false);
                sidecar.status.setLastError("rerank.backend=" + rerankMode.token()
                        + " requested but " + rerankMissingMsg.toLowerCase(java.util.Locale.ROOT));
                System.exit(3);
                return;
            }
            log.info("{} – rerank handler will report not-implemented", rerankMissingMsg);
        } else {
            Reranker rr = null;
            String rerankBackend = null;
            Exception forcedFailure = null;
            String autoFallbackMsg = null;
            try {
                if (rerankMode == RerankerBackendMode.DIRECTML
                        || rerankMode == RerankerBackendMode.AUTO) {
                    try {
                        rr = BertCrossEncoderRerankers.loadDirectMl(rerankDir);
                        rerankBackend = "directml";
                    } catch (Exception gpuFail) {
                        if (rerankMode == RerankerBackendMode.DIRECTML) {
                            forcedFailure = gpuFail;
                            throw gpuFail;
                        }
                        log.warn("Reranker DirectML init failed ({}), falling back to CPU",
                                gpuFail.getMessage());
                        autoFallbackMsg = "rerank.backend=auto fell back to cpu: "
                                + gpuFail.getMessage();
                    }
                }
                if (rr == null && (rerankMode == RerankerBackendMode.CPU
                        || rerankMode == RerankerBackendMode.AUTO)) {
                    try {
                        rr = BertCrossEncoderRerankers.loadCpu(rerankDir);
                        rerankBackend = "cpu";
                    } catch (Exception cpuFail) {
                        if (rerankMode == RerankerBackendMode.CPU) {
                            forcedFailure = cpuFail;
                            throw cpuFail;
                        }
                        throw cpuFail;
                    }
                }
                sidecar.withRerankerBackend(rerankBackend, rr);
                if (autoFallbackMsg != null) {
                    // Surface the auto-fallback in the health endpoint so the
                    // workbench shows *why* we are on CPU even though DirectML
                    // is available on the box. Use the dedicated fallback
                    // signal – lastError stays reserved for hard errors.
                    sidecar.status.setRerankerFallback(true);
                    sidecar.status.setRerankerFallbackReason(autoFallbackMsg);
                }
                log.info("Reranker backend ready: mode={} backend={} modelDir={}",
                        rerankMode.token(), rerankBackend, rerankDir);
            } catch (Exception e) {
                log.error("Reranker initialisation failed (mode={}): {}",
                        rerankMode.token(), e.getMessage(), e);
                sidecar.status.setRerankerBackend("error");
                sidecar.status.setRerankerReady(false);
                sidecar.status.setLastError("rerank.backend=" + rerankMode.token()
                        + " failed: " + e.getMessage());
                if (forcedFailure != null
                        || rerankMode == RerankerBackendMode.CPU
                        || rerankMode == RerankerBackendMode.DIRECTML) {
                    System.exit(3);
                    return;
                }
            }
        }

        int exitCode = sidecar.run();
        System.exit(exitCode);
    }

    /**
     * Look up the reranker model directory.
     * Honours {@code -Drerank.modelDir} first; otherwise probes a small
     * list of conventional locations under {@code model/}.
     */
    private static Path resolveRerankerDir() {
        String override = System.getProperty("rerank.modelDir");
        if (override != null && !override.isBlank()) {
            Path p = Path.of(override);
            return Files.exists(p.resolve("model.safetensors"))
                    && Files.exists(p.resolve("tokenizer.json"))
                    && Files.exists(p.resolve("config.json")) ? p : null;
        }
        for (Path candidate : new Path[]{
                Path.of("model/cross-encoder-ms-marco-MiniLM-L-6-v2"),
                Path.of("model/cross-encoder/ms-marco-MiniLM-L-6-v2"),
                Path.of("model/ms-marco-MiniLM-L-6-v2"),
        }) {
            if (Files.exists(candidate.resolve("model.safetensors"))
                    && Files.exists(candidate.resolve("tokenizer.json"))
                    && Files.exists(candidate.resolve("config.json"))) {
                return candidate;
            }
        }
        return null;
    }

    private static Path resolveMiniLmDir() {
        String override = System.getProperty("minilm.modelDir");
        if (override != null && !override.isBlank()) {
            Path p = Path.of(override);
            return Files.exists(p.resolve("model.safetensors")) ? p : null;
        }
        for (Path candidate : new Path[]{
                Path.of("model/all-MiniLM-L6-v2"),
                Path.of("model/sentence-transformers/all-MiniLM-L6-v2"),
        }) {
            if (Files.exists(candidate.resolve("model.safetensors"))
                    && Files.exists(candidate.resolve("tokenizer.json"))) {
                return candidate;
            }
        }
        return null;
    }

    /**
     * Parse {@code -Dembed.model} into a stable family token. Defaults
     * to {@code minilm}; supported: {@code minilm}, {@code e5}.
     *
     * <p>The parser also accepts the full Huggingface-style model IDs
     * from the {@link EmbeddingModelRegistry}. Non-embedding IDs
     * (decoder / summarizer) are rejected with the explicit
     * "{@code … is not an embedding model …}" message; embedding IDs
     * that are classified as {@code planned} / unimplemented are
     * rejected with a registry-driven status message.
     */
    static String embedFamily(String raw) {
        if (raw == null || raw.isBlank()) return "minilm";
        String s = raw.trim().toLowerCase(java.util.Locale.ROOT);

        // Registry-driven gate first: full model IDs from the company
        // model list. This is where decoder / summarizer IDs get
        // rejected with a clear use-case-specific message instead of
        // falling through to the generic "Unknown embed.model" error.
        EmbeddingModelRegistry.Entry known = EmbeddingModelRegistry.findByModelId(s);
        if (known != null) {
            if (!known.isEmbedding()) {
                throw new IllegalArgumentException(
                        EmbeddingModelRegistry.nonEmbeddingErrorMessage(known));
            }
            if (known.embedFamily() == null) {
                throw new IllegalArgumentException(
                        EmbeddingModelRegistry.unimplementedEmbeddingErrorMessage(known));
            }
            return known.embedFamily();
        }

        return switch (s) {
            case "minilm", "mini-lm", "minilm-l6", "all-minilm-l6-v2" -> "minilm";
            case "e5", "e5-base", "e5-base-sts", "e5-base-sts-en-de" -> "e5";
            default -> throw new IllegalArgumentException(
                    "Unknown embed.model: '" + raw + "' (supported: minilm, e5)");
        };
    }

    private static Path resolveE5Dir(E5Variant variant) {
        String override = System.getProperty("e5.modelDir");
        if (override != null && !override.isBlank()) {
            Path p = Path.of(override);
            return Files.exists(p.resolve("model.safetensors"))
                    && Files.exists(p.resolve("tokenizer.json")) ? p : null;
        }
        for (Path candidate : variant.directoryHints()) {
            if (Files.exists(candidate.resolve("model.safetensors"))
                    && Files.exists(candidate.resolve("tokenizer.json"))) {
                return candidate;
            }
        }
        return null;
    }

    private static Path resolveModelDir() {
        String override = System.getProperty("phi3.modelDir");
        if (override != null && !override.isBlank()) {
            return Path.of(override);
        }
        Path rel = Path.of("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(rel.resolve("model.onnx"))) return rel;
        Path parent = Path.of(System.getProperty("user.dir")).getParent()
                .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(parent.resolve("model.onnx"))) return parent;
        return rel;
    }

    // ── Main loop ────────────────────────────────────────────────────────

    public int run() {
        log.info("DirectMlPhi3Sidecar starting (modelDir={}, backend={}, autoLoad={})",
                modelDir, backend, autoLoadModel);

        try (JsonRpcMessageReader reader = new JsonRpcMessageReader(in, mapper);
             JsonRpcMessageWriter writer = new JsonRpcMessageWriter(out, mapper)) {

            // Probe the Phi-3 model directory *before* constructing a
            // Phi3Summarizer. When the directory is missing or
            // incomplete we deliberately do not register a real
            // SummarizeHandler – the fallback handler then answers
            // `summarize` with -32005 NOT_IMPLEMENTED, matching the
            // embed / rerank "no model" semantics and the documented
            // contract in README / PROTOCOL / SUPPORTED_MODELS. The
            // specific missing file is still surfaced via
            // `sidecar.modelLoadFailed` + `status.lastError` so the
            // workbench can show it.
            String missing = autoLoadModel && summarizer == null
                    ? Phi3InferenceEngine.describeMissingModelFile(modelDir)
                    : null;

            if (summarizer == null && autoLoadModel && missing == null) {
                ownedSummarizer = new Phi3Summarizer(modelDir, defaultMaxTokens, backend);
                summarizer = ownedSummarizer;
            }
            registerHandlers();

            if (autoLoadModel && ownedSummarizer != null) {
                Thread loader = new Thread(() -> loadModel(writer), "phi3-model-loader");
                loader.setDaemon(true);
                loader.start();
            } else if (missing != null) {
                log.warn("{} – summarize will respond with -32005 NOT_IMPLEMENTED", missing);
                status.setLastError(missing);
                writer.writeNotification(JsonRpcNotification.of("sidecar.modelLoadFailed",
                        Map.of("error", missing, "modelDir", String.valueOf(modelDir))));
            }

            writer.writeNotification(JsonRpcNotification.of("sidecar.started", started()));

            RawLine raw;
            while (!status.isShuttingDown() && (raw = reader.readNext()) != null) {
                if (raw.hasError()) {
                    writer.writeResponse(JsonRpcResponse.failure(NullNode.getInstance(),
                            new JsonRpcError(JsonRpcErrorCode.PARSE_ERROR,
                                    "Parse error: " + raw.parseError().getMessage())));
                    continue;
                }
                JsonRpcRequest request = raw.request();
                // Pass writer so async handlers (e.g. SummarizeHandler) can respond
                // from their own thread without blocking this dispatch loop.
                JsonRpcResponse response = dispatcher.dispatch(request, writer);
                if (!request.isNotification() && response != null) {
                    // response == null → async handler already wrote the response
                    writer.writeResponse(response);
                }
                if (status.isShuttingDown()) break;
            }

            log.info("Sidecar shutting down (stdin closed or shutdown requested)");

            // Wait briefly for in-flight async handlers (e.g. SummarizeHandler,
            // which offloads inference to a daemon worker thread) to finish
            // writing their response before the writer is closed by
            // try-with-resources. Without this wait, a `summarize` immediately
            // followed by `shutdown` is racy: the dispatch loop may exit and
            // close stdout before the async worker writes its response. The
            // async handlers set `status.setBusy(true)` synchronously on the
            // dispatch thread before starting their worker, so by the time we
            // reach this point `isBusy()` reliably reflects pending async work.
            // Bounded so a hung worker can't block shutdown indefinitely.
            long deadlineNanos = System.nanoTime()
                    + java.util.concurrent.TimeUnit.SECONDS.toNanos(5);
            while (status.isBusy() && System.nanoTime() < deadlineNanos) {
                try {
                    Thread.sleep(5);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        } catch (IOException e) {
            log.error("Sidecar IO error", e);
            return 1;
        } finally {
            if (ownedSummarizer != null) {
                try {
                    ownedSummarizer.shutdown();
                } catch (Exception e) {
                    log.warn("Error during summarizer shutdown: {}", e.getMessage());
                }
            }
        }
        return 0;
    }

    private void registerHandlers() {
        dispatcher.register("health", new HealthHandler(status));
        if (summarizer != null) {
            dispatcher.register("summarize", new SummarizeHandler(summarizer, status));
        } else {
            // No Phi-3 summarizer was configured at all (model directory
            // missing or autoLoadModel=false in tests). Match the embed /
            // rerank fallback pattern: report NOT_IMPLEMENTED so clients
            // can degrade gracefully instead of seeing a stack trace.
            dispatcher.register("summarize", params -> {
                throw new com.aresstack.windirectml.sidecar.jsonrpc.JsonRpcMethodException(
                        JsonRpcErrorCode.NOT_IMPLEMENTED,
                        "summarize not implemented: no Phi-3 summarizer configured "
                                + "(summarizer support is experimental and Phi-3-only)");
            });
        }
        dispatcher.register("embed", new EmbedHandler(embeddingModel::get, status));
        dispatcher.register("embedBatch", new EmbedBatchHandler(embeddingModel::get, status));
        dispatcher.register("rerank", new RerankHandler(reranker::get, status));
        dispatcher.register("shutdown", new ShutdownHandler(status));
        dispatcher.register("cancel", new CancelHandler());
    }

    private void loadModel(JsonRpcMessageWriter writer) {
        try {
            log.info("Loading Phi-3 model from {} (backend={})", modelDir, backend);
            String missing = Phi3InferenceEngine.describeMissingModelFile(modelDir);
            if (missing != null) {
                log.error("{}", missing);
                status.setLastError(missing);
                writer.writeNotification(JsonRpcNotification.of("sidecar.modelLoadFailed",
                        Map.of("error", missing, "modelDir", String.valueOf(modelDir))));
                return;
            }
            long t0 = System.currentTimeMillis();
            ownedSummarizer.initialize();
            long elapsed = System.currentTimeMillis() - t0;
            status.setModelLoaded(true);
            status.setMode("phi-3 (" + backend + ")");
            status.setSummarizerReady(true);
            status.setSummarizerBackend(backend);
            status.setSummarizerModel("phi-3-mini-int4-" + backend);
            Map<String, Object> params = new LinkedHashMap<>();
            params.put("loadTimeMs", elapsed);
            params.put("backend", backend);
            params.put("modelDir", modelDir.toString());
            writer.writeNotification(JsonRpcNotification.of("sidecar.modelLoaded", params));
            log.info("Phi-3 model loaded in {} ms", elapsed);
        } catch (Throwable t) {
            log.error("Model load failed", t);
            String msg = t.getMessage() != null ? t.getMessage() : t.getClass().getName();
            status.setLastError(msg);
            status.setSummarizerReady(false);
            writer.writeNotification(JsonRpcNotification.of("sidecar.modelLoadFailed",
                    Map.of("error", msg, "modelDir", String.valueOf(modelDir))));
        }
    }

    private Map<String, Object> started() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", "DirectMlPhi3Sidecar");
        m.put("protocol", "jsonrpc-2.0");
        m.put("methods", new String[]{"health", "summarize", "embed", "embedBatch", "rerank", "shutdown", "cancel"});
        m.put("backend", backend);
        m.put("modelDir", modelDir.toString());
        m.put("autoLoadModel", autoLoadModel);
        m.put("embeddingBackend",
                status.getEmbeddingBackend() != null ? status.getEmbeddingBackend() : "none");
        m.put("rerankerBackend",
                status.getRerankerBackend() != null ? status.getRerankerBackend() : "none");
        return m;
    }
}
