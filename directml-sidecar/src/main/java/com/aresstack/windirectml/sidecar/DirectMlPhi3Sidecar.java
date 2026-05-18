package com.aresstack.windirectml.sidecar;

import com.aresstack.windirectml.inference.Phi3InferenceEngine;
import com.aresstack.windirectml.inference.Phi3Summarizer;
import com.aresstack.windirectml.sidecar.handlers.CancelHandler;
import com.aresstack.windirectml.sidecar.handlers.EmbedHandler;
import com.aresstack.windirectml.sidecar.handlers.HealthHandler;
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
 * <p>
 * The model is loaded asynchronously on a background thread so {@code health}
 * is answerable immediately after process start.
 */
public final class DirectMlPhi3Sidecar {

    private static final Logger log = LoggerFactory.getLogger(DirectMlPhi3Sidecar.class);

    private final InputStream in;
    private final OutputStream out;
    private final Path modelDir;
    private final String backend;
    private final int defaultMaxTokens;

    private final ObjectMapper mapper = new ObjectMapper();
    private final SidecarStatus status = new SidecarStatus();
    private final SidecarCommandDispatcher dispatcher = new SidecarCommandDispatcher();

    private Phi3Summarizer summarizer;

    public DirectMlPhi3Sidecar(InputStream in, OutputStream out, Path modelDir,
                               String backend, int defaultMaxTokens) {
        this.in = in;
        this.out = out;
        this.modelDir = modelDir;
        this.backend = backend != null ? backend : "auto";
        this.defaultMaxTokens = defaultMaxTokens > 0 ? defaultMaxTokens : 512;
    }

    // ── Entry point ──────────────────────────────────────────────────────

    public static void main(String[] args) {
        Path modelDir = resolveModelDir();
        String backend = System.getProperty("phi3.backend", "auto");
        int maxTokens = Integer.getInteger("phi3.maxTokens", 512);

        DirectMlPhi3Sidecar sidecar = new DirectMlPhi3Sidecar(
                System.in, System.out, modelDir, backend, maxTokens);
        int exitCode = sidecar.run();
        System.exit(exitCode);
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
        log.info("DirectMlPhi3Sidecar starting (modelDir={}, backend={})", modelDir, backend);

        try (JsonRpcMessageReader reader = new JsonRpcMessageReader(in, mapper);
             JsonRpcMessageWriter writer = new JsonRpcMessageWriter(out, mapper)) {

            // Register protocol handlers BEFORE model load so health works immediately.
            summarizer = new Phi3Summarizer(modelDir, defaultMaxTokens, backend);
            registerHandlers();

            // Kick off async model load.
            Thread loader = new Thread(() -> loadModel(writer), "phi3-model-loader");
            loader.setDaemon(true);
            loader.start();

            // Emit "started" notification right away.
            writer.writeNotification(JsonRpcNotification.of("sidecar.started", started()));

            // Dispatch loop.
            RawLine raw;
            while (!status.isShuttingDown() && (raw = reader.readNext()) != null) {
                if (raw.hasError()) {
                    writer.writeResponse(JsonRpcResponse.failure(NullNode.getInstance(),
                            new JsonRpcError(JsonRpcErrorCode.PARSE_ERROR,
                                    "Parse error: " + raw.parseError().getMessage())));
                    continue;
                }
                JsonRpcRequest request = raw.request();
                JsonRpcResponse response = dispatcher.dispatch(request);
                if (!request.isNotification()) {
                    writer.writeResponse(response);
                }
                if (status.isShuttingDown()) break;
            }

            log.info("Sidecar shutting down (stdin closed or shutdown requested)");
        } catch (IOException e) {
            log.error("Sidecar IO error", e);
            return 1;
        } finally {
            if (summarizer != null) {
                try {
                    summarizer.shutdown();
                } catch (Exception e) {
                    log.warn("Error during summarizer shutdown: {}", e.getMessage());
                }
            }
        }
        return 0;
    }

    private void registerHandlers() {
        dispatcher.register("health", new HealthHandler(status));
        dispatcher.register("summarize", new SummarizeHandler(summarizer, status));
        dispatcher.register("embed", new EmbedHandler());
        dispatcher.register("shutdown", new ShutdownHandler(status));
        dispatcher.register("cancel", new CancelHandler());
    }

    private void loadModel(JsonRpcMessageWriter writer) {
        try {
            log.info("Loading Phi-3 model from {} (backend={})", modelDir, backend);
            if (!Phi3InferenceEngine.isValidModelDir(modelDir)) {
                String msg = "Model directory invalid or incomplete: " + modelDir;
                log.error(msg);
                status.setLastError(msg);
                writer.writeNotification(JsonRpcNotification.of("sidecar.modelLoadFailed",
                        Map.of("error", msg, "modelDir", modelDir.toString())));
                return;
            }
            long t0 = System.currentTimeMillis();
            summarizer.initialize();
            long elapsed = System.currentTimeMillis() - t0;
            status.setModelLoaded(true);
            status.setMode("phi-3 (" + backend + ")");
            Map<String, Object> params = new LinkedHashMap<>();
            params.put("loadTimeMs", elapsed);
            params.put("backend", backend);
            params.put("modelDir", modelDir.toString());
            writer.writeNotification(JsonRpcNotification.of("sidecar.modelLoaded", params));
            log.info("Phi-3 model loaded in {} ms", elapsed);
        } catch (Throwable t) {
            log.error("Model load failed", t);
            status.setLastError(t.getMessage());
            writer.writeNotification(JsonRpcNotification.of("sidecar.modelLoadFailed",
                    Map.of("error", String.valueOf(t.getMessage()))));
        }
    }

    private Map<String, Object> started() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", "DirectMlPhi3Sidecar");
        m.put("protocol", "jsonrpc-2.0");
        m.put("methods", new String[]{"health", "summarize", "embed", "shutdown", "cancel"});
        m.put("backend", backend);
        m.put("modelDir", modelDir.toString());
        return m;
    }
}

