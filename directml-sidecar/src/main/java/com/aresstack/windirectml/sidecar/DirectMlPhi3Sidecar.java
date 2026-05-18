package com.aresstack.windirectml.inference;

import com.aresstack.windirectml.inference.phi3.Phi3Config;
import com.aresstack.windirectml.inference.phi3.Phi3GpuKernels;
import com.aresstack.windirectml.inference.phi3.Phi3GpuPipeline;
import com.aresstack.windirectml.inference.phi3.Phi3Runtime;
import com.aresstack.windirectml.inference.phi3.Phi3Tokenizer;
import com.aresstack.windirectml.inference.phi3.Phi3Tokenizer.ChatMessage;
import com.aresstack.windirectml.inference.phi3.Phi3Weights;
import com.aresstack.windirectml.windows.WindowsBindings;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * JSON-based CLI for testing the Phi-3 model interactively from a terminal.
 * <p>
 * Designed to be driven by an AI agent or human tester.  Every output line is
 * valid JSON so the caller can parse responses programmatically.
 * <p>
 * <h3>System commands (start with {@code /}):</h3>
 * <pre>
 *   /help                – show available commands
 *   /status              – show model status (loaded, mode, config)
 *   /maxTokens &lt;n&gt;       – set default max tokens for generation
 *   /systemPrompt &lt;text&gt; – set or clear the system prompt
 *   /history             – show conversation history
 *   /clear               – clear conversation history
 *   /exit                – quit the CLI
 * </pre>
 * <p>
 * <h3>Chat messages (JSON on a single line):</h3>
 * <pre>
 * {"prompt":"Hello, how are you?"}
 * {"prompt":"What is 2+2?","maxTokens":32}
 * {"prompt":"Explain gravity","maxTokens":128,"systemPrompt":"You are a physics professor."}
 * </pre>
 * <p>
 * <h3>Responses (JSON, one per line):</h3>
 * <pre>
 * {"type":"response","text":"...","tokens":42,"timings":{...},"profile":"..."}
 * {"type":"status","message":"Model loaded","mode":"GPU (32/32 layers)","config":{...}}
 * {"type":"error","message":"..."}
 * {"type":"system","command":"/help","result":"..."}
 * </pre>
 * <p>
 * Start with:
 * <pre>
 *   --enable-native-access=ALL-UNNAMED -Xmx4g
 * </pre>
 */
public class DirectMlPhi3Sidecar {

    // ── Model path ───────────────────────────────────────────────────────
    private static final Path MODEL_DIR = resolveModelDir();

    private static Path resolveModelDir() {
        Path rel = Path.of("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(rel.resolve("model.onnx"))) return rel;

        Path parent = Path.of(System.getProperty("user.dir")).getParent()
                .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(parent.resolve("model.onnx"))) return parent;

        return rel;
    }

    // ── State ────────────────────────────────────────────────────────────
    private Phi3Config config;
    private Phi3Tokenizer tokenizer;
    private Phi3Weights weights;
    private Phi3Runtime runtime;
    private WindowsBindings wb;
    private Phi3GpuKernels gpuKernels;
    private Phi3GpuPipeline gpuPipeline;
    private volatile boolean modelReady = false;
    private String mode = "unknown";

    private int defaultMaxTokens = 0; // 0 = unlimited (until EOS or context limit)
    private String systemPrompt =
            "You are a helpful AI assistant. Answer concisely and accurately. " +
            "Respond in the same language the user writes in.";

    private final List<ChatMessage> conversationHistory = new ArrayList<>();

    // ── Output writer (always stdout, one JSON per line) ─────────────────
    private final PrintWriter out;
    private final boolean skipModelLoading;

    // ══════════════════════════════════════════════════════════════════════

    public DirectMlPhi3Sidecar(PrintWriter out) {
        this(out, false);
    }

    /**
     * @param skipModelLoading if {@code true}, skip model loading (for unit tests)
     */
    DirectMlPhi3Sidecar(PrintWriter out, boolean skipModelLoading) {
        this.out = out;
        this.skipModelLoading = skipModelLoading;
    }

    public static void main(String[] args) {
        PrintWriter pw = new PrintWriter(new BufferedOutputStream(System.out), true);
        DirectMlPhi3Sidecar cli = new DirectMlPhi3Sidecar(pw);
        cli.run(new Scanner(System.in));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Main loop
    // ══════════════════════════════════════════════════════════════════════

    public void run(Scanner scanner) {
        emitSystem("ready", "DirectMlPhi3Sidecar started. Loading model from: " + MODEL_DIR.toAbsolutePath());
        if (!skipModelLoading) {
            loadModel();
        } else {
            emitSystem("skip", "Model loading skipped (test mode).");
        }

        while (scanner.hasNextLine()) {
            String line = scanner.nextLine().trim();
            if (line.isEmpty()) continue;

            if (line.startsWith("/")) {
                if (!handleCommand(line)) break;  // /exit returns false
            } else {
                handleJsonMessage(line);
            }
        }

        cleanup();
        emitSystem("exit", "Bye.");
    }

    // ══════════════════════════════════════════════════════════════════════
    // System commands
    // ══════════════════════════════════════════════════════════════════════

    /**
     * @return false if the CLI should exit
     */
    private boolean handleCommand(String line) {
        String[] parts = line.split("\\s+", 2);
        String cmd = parts[0].toLowerCase();
        String arg = parts.length > 1 ? parts[1].trim() : "";

        switch (cmd) {
            case "/exit", "/quit" -> {
                return false;
            }
            case "/help" -> {
                String maxDesc = defaultMaxTokens <= 0 ? "unlimited" : String.valueOf(defaultMaxTokens);
                String help = """
                        Available commands:
                          /help                – show this help
                          /status              – show model status
                          /maxTokens <n|0>     – set default max tokens (0 = unlimited, current: %s)
                          /systemPrompt <text> – set system prompt (empty = clear)
                          /history             – show conversation history
                          /clear               – clear conversation history
                          /exit                – quit the CLI
                        
                        Chat via JSON (one line):
                          {"prompt":"Your question here"}
                          {"prompt":"...","maxTokens":128}
                          {"prompt":"...","maxTokens":0}   ← unlimited (until EOS)
                        """.formatted(maxDesc);
                emitCommandResult("/help", help.strip());
            }
            case "/status" -> {
                Map<String, Object> status = new LinkedHashMap<>();
                status.put("modelReady", modelReady);
                status.put("mode", mode);
                status.put("defaultMaxTokens", defaultMaxTokens <= 0 ? "unlimited" : defaultMaxTokens);
                status.put("systemPrompt", systemPrompt);
                status.put("modelDir", MODEL_DIR.toAbsolutePath().toString());
                if (config != null) {
                    Map<String, Object> cfg = new LinkedHashMap<>();
                    cfg.put("hiddenSize", config.hiddenSize());
                    cfg.put("numHiddenLayers", config.numHiddenLayers());
                    cfg.put("vocabSize", config.vocabSize());
                    cfg.put("maxPositionEmbeddings", config.maxPositionEmbeddings());
                    status.put("config", cfg);
                }
                emitJson(mapOf("type", "status", "data", status));
            }
            case "/maxtokens" -> {
                try {
                    int n = Integer.parseInt(arg);
                    if (n < 0 || n > 4096) throw new NumberFormatException("out of range");
                    defaultMaxTokens = n;
                    String desc = n == 0 ? "unlimited (until EOS)" : String.valueOf(n);
                    emitCommandResult("/maxTokens", "maxTokens set to " + desc);
                } catch (NumberFormatException e) {
                    emitError("Invalid maxTokens value: '" + arg + "'. Use: /maxTokens <1-4096 or 0 for unlimited>");
                }
            }
            case "/systemprompt" -> {
                systemPrompt = arg.isEmpty() ? null : arg;
                emitCommandResult("/systemPrompt",
                        systemPrompt == null ? "System prompt cleared." : "System prompt set.");
            }
            case "/history" -> {
                List<Map<String, String>> historyList = new ArrayList<>();
                for (ChatMessage msg : conversationHistory) {
                    historyList.add(mapOf("role", msg.role(), "content", msg.content()));
                }
                emitJson(mapOf("type", "history", "messages", historyList));
            }
            case "/clear" -> {
                conversationHistory.clear();
                emitCommandResult("/clear", "Conversation history cleared.");
            }
            case "/benchmark" -> {
                runBenchmark();
            }
            default -> emitError("Unknown command: " + cmd + ". Type /help for available commands.");
        }
        return true;
    }

    // ══════════════════════════════════════════════════════════════════════
    // JSON chat messages
    // ══════════════════════════════════════════════════════════════════════

    @SuppressWarnings("unchecked")
    private void handleJsonMessage(String line) {
        if (!modelReady) {
            emitError("Model not loaded yet. Please wait for the 'model_loaded' status message.");
            return;
        }

        // Minimal JSON parsing (avoids extra dependency in test scope).
        // We expect: {"prompt":"...","maxTokens":N,"systemPrompt":"..."}
        Map<String, Object> request;
        try {
            request = parseSimpleJson(line);
        } catch (Exception e) {
            emitError("Invalid JSON input: " + e.getMessage()
                    + ". Expected: {\"prompt\":\"...\"}  or a /command.");
            return;
        }

        String prompt = stringVal(request, "prompt");
        if (prompt == null || prompt.isBlank()) {
            emitError("Missing or empty 'prompt' field in JSON.");
            return;
        }

        int maxTokens = intVal(request, "maxTokens", defaultMaxTokens);
        // 0 means "unlimited" — generate until EOS (up to model context limit)
        if (maxTokens <= 0 && config != null) {
            maxTokens = config.maxPositionEmbeddings();
        } else if (maxTokens <= 0) {
            maxTokens = 4096;
        }
        String sysPrompt = stringVal(request, "systemPrompt");
        if (sysPrompt == null) sysPrompt = systemPrompt;

        // Build multi-turn prompt (user message added to history AFTER successful generation)
        List<ChatMessage> pendingHistory = new ArrayList<>(conversationHistory);
        pendingHistory.add(ChatMessage.user(prompt));

        try {
            String formatted = tokenizer.formatMultiTurnChat(sysPrompt, pendingHistory);

            long t0 = System.nanoTime();
            final int[] tokenCount = {0};
            final StringBuilder responseBuffer = new StringBuilder();

            String response = runtime.generateStreaming(formatted, maxTokens,
                    (tokenId, textSoFar, delta) -> {
                        tokenCount[0]++;
                        responseBuffer.setLength(0);
                        responseBuffer.append(textSoFar);
                    });

            long elapsedNs = System.nanoTime() - t0;
            double elapsedMs = elapsedNs / 1e6;
            double msPerToken = tokenCount[0] > 0 ? elapsedMs / tokenCount[0] : 0;
            double tokensPerSec = (tokenCount[0] > 0 && elapsedMs > 0)
                    ? tokenCount[0] / (elapsedMs / 1000.0) : 0;

            // Record both messages only after successful generation
            conversationHistory.add(ChatMessage.user(prompt));
            conversationHistory.add(ChatMessage.assistant(response));

            // Build timing info
            Map<String, Object> timings = new LinkedHashMap<>();
            timings.put("totalMs", round2(elapsedMs));
            timings.put("tokens", tokenCount[0]);
            timings.put("msPerToken", round2(msPerToken));
            timings.put("tokensPerSec", round2(tokensPerSec));

            // Get detailed profile
            String profile = runtime.getLastProfile();

            // Emit response
            Map<String, Object> resp = new LinkedHashMap<>();
            resp.put("type", "response");
            resp.put("text", response);
            resp.put("tokens", tokenCount[0]);
            resp.put("timings", timings);
            if (profile != null) resp.put("profile", profile);
            resp.put("timestamp", Instant.now().toString());

            emitJson(resp);

        } catch (Exception e) {
            emitError("Generation failed: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model loading
    // ══════════════════════════════════════════════════════════════════════

    private void loadModel() {
        long t0 = System.currentTimeMillis();
        try {
            if (!Phi3InferenceEngine.isValidModelDir(MODEL_DIR)) {
                emitError("Model not found in: " + MODEL_DIR.toAbsolutePath()
                        + ". Ensure config.json, tokenizer.json, model.onnx and model.onnx.data are present.");
                return;
            }

            config = Phi3Config.load(MODEL_DIR.resolve("config.json"));
            tokenizer = Phi3Tokenizer.load(MODEL_DIR.resolve("tokenizer.json"));
            weights = Phi3Weights.load(MODEL_DIR, config);

            // ── Try GPU acceleration ─────────────────────────────────
            mode = "CPU";
            try {
                if (WindowsBindings.isSupported()) {
                    wb = new WindowsBindings();
                    wb.init("directml");
                    if (wb.hasDirectMl()) {
                        int gpuLayers = Integer.getInteger("phi3.gpu.layers",
                                config.numHiddenLayers());
                        boolean gpuLmHead = Boolean.parseBoolean(
                                System.getProperty("phi3.gpu.lmhead", "true"));
                        gpuKernels = Phi3GpuKernels.create(
                                wb, weights, config, gpuLayers, gpuLmHead);
                        gpuPipeline = new Phi3GpuPipeline(wb, gpuKernels, config);
                        gpuPipeline.uploadLayerWeights(wb, weights, config);
                        mode = "GPU V2.0 (" + gpuKernels.getGpuLayers() + "/"
                                + config.numHiddenLayers() + " layers, pipeline)";
                    }
                }
            } catch (Exception gpuEx) {
                System.err.println("GPU init failed, falling back to CPU: " + gpuEx.getMessage());
                gpuEx.printStackTrace(System.err);
                if (gpuPipeline != null) { try { gpuPipeline.close(); } catch (Exception ignored) {} gpuPipeline = null; }
                if (gpuKernels != null) { try { gpuKernels.close(); } catch (Exception ignored) {} gpuKernels = null; }
                if (wb != null) { try { wb.close(); } catch (Exception ignored) {} wb = null; }
            }

            runtime = new Phi3Runtime(config, weights, tokenizer, gpuKernels, gpuPipeline);

            long elapsed = System.currentTimeMillis() - t0;
            modelReady = true;

            Map<String, Object> loaded = new LinkedHashMap<>();
            loaded.put("type", "status");
            loaded.put("event", "model_loaded");
            loaded.put("mode", mode);
            loaded.put("loadTimeMs", elapsed);
            loaded.put("hiddenSize", config.hiddenSize());
            loaded.put("numLayers", config.numHiddenLayers());
            loaded.put("vocabSize", config.vocabSize());
            emitJson(loaded);

        } catch (Exception e) {
            emitError("Failed to load model: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    private void cleanup() {
        if (gpuPipeline != null) try { gpuPipeline.close(); } catch (Exception ignored) {}
        if (gpuKernels != null) try { gpuKernels.close(); } catch (Exception ignored) {}
        if (wb != null) try { wb.close(); } catch (Exception ignored) {}
    }

    // ══════════════════════════════════════════════════════════════════════
    // Benchmark
    // ══════════════════════════════════════════════════════════════════════

    private void runBenchmark() {
        if (!modelReady) {
            emitError("Model not loaded. Cannot benchmark.");
            return;
        }

        emitSystem("benchmark", "Starting GPU pipeline benchmark...");

        String warmupPrompt = "<|system|>\nYou are helpful.<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\n";
        String benchPrompt = "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\nExplain the difference between TCP and UDP in networking.<|end|>\n<|assistant|>\n";

        try {
            // Warmup: 2 short generations
            for (int i = 0; i < 2; i++) {
                runtime.generateStreaming(warmupPrompt, 16, null);
            }

            // Benchmark: 3 runs of 32 tokens each
            int benchTokens = 32;
            int runs = 3;
            double[] msPerToken = new double[runs];
            String[] profiles = new String[runs];

            for (int r = 0; r < runs; r++) {
                runtime.resetCache();
                long t0 = System.nanoTime();
                final int[] tokenCount = {0};
                runtime.generateStreaming(benchPrompt, benchTokens,
                        (id, text, delta) -> tokenCount[0]++);
                long elapsed = System.nanoTime() - t0;
                msPerToken[r] = (elapsed / 1e6) / Math.max(tokenCount[0], 1);
                profiles[r] = runtime.getLastProfile();
            }

            // Report
            double avg = 0;
            for (double v : msPerToken) avg += v;
            avg /= runs;
            double min = msPerToken[0], max = msPerToken[0];
            for (double v : msPerToken) { min = Math.min(min, v); max = Math.max(max, v); }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("type", "benchmark");
            result.put("mode", mode);
            result.put("runs", runs);
            result.put("tokensPerRun", benchTokens);
            result.put("avgMsPerToken", round2(avg));
            result.put("minMsPerToken", round2(min));
            result.put("maxMsPerToken", round2(max));
            result.put("avgTokensPerSec", round2(1000.0 / avg));
            result.put("profile", profiles[runs - 1]);
            emitJson(result);

        } catch (Exception e) {
            emitError("Benchmark failed: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // JSON output helpers
    // ══════════════════════════════════════════════════════════════════════

    private void emitJson(Map<String, Object> map) {
        out.println(toJson(map));
        out.flush();
    }

    private void emitSystem(String event, String message) {
        emitJson(mapOf("type", "system", "event", event, "message", message));
    }

    private void emitCommandResult(String command, String result) {
        emitJson(mapOf("type", "command", "command", command, "result", result));
    }

    private void emitError(String message) {
        emitJson(mapOf("type", "error", "message", message));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Minimal JSON serializer (no external dependencies in test scope)
    // ══════════════════════════════════════════════════════════════════════

    @SuppressWarnings("unchecked")
    private static String toJson(Object obj) {
        if (obj == null) return "null";
        if (obj instanceof String s) return "\"" + escapeJsonString(s) + "\"";
        if (obj instanceof Number n) return n.toString();
        if (obj instanceof Boolean b) return b.toString();
        if (obj instanceof Map<?, ?> m) {
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(",");
                first = false;
                sb.append("\"").append(escapeJsonString(e.getKey().toString())).append("\":");
                sb.append(toJson(e.getValue()));
            }
            sb.append("}");
            return sb.toString();
        }
        if (obj instanceof List<?> list) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(",");
                sb.append(toJson(list.get(i)));
            }
            sb.append("]");
            return sb.toString();
        }
        return "\"" + escapeJsonString(obj.toString()) + "\"";
    }

    private static String escapeJsonString(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"'  -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Minimal JSON parser (handles flat objects with string/number values)
    // ══════════════════════════════════════════════════════════════════════

    private static Map<String, Object> parseSimpleJson(String json) {
        json = json.trim();
        if (!json.startsWith("{") || !json.endsWith("}")) {
            throw new IllegalArgumentException("Expected JSON object {...}");
        }
        json = json.substring(1, json.length() - 1).trim();

        Map<String, Object> map = new LinkedHashMap<>();
        int i = 0;
        while (i < json.length()) {
            // skip whitespace and commas
            while (i < json.length() && (json.charAt(i) == ',' || Character.isWhitespace(json.charAt(i)))) i++;
            if (i >= json.length()) break;

            // parse key
            if (json.charAt(i) != '"') throw new IllegalArgumentException("Expected '\"' at position " + i);
            String key = parseJsonString(json, i);
            i += key.length() + 2; // skip quotes
            // un-escape
            key = unescapeJsonString(key);

            // skip colon
            while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
            if (i >= json.length() || json.charAt(i) != ':')
                throw new IllegalArgumentException("Expected ':' at position " + i);
            i++;
            while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;

            // parse value
            if (i >= json.length()) throw new IllegalArgumentException("Unexpected end of input");
            char c = json.charAt(i);
            if (c == '"') {
                String val = parseJsonString(json, i);
                i += val.length() + 2;
                map.put(key, unescapeJsonString(val));
            } else if (c == '-' || Character.isDigit(c)) {
                int start = i;
                boolean isFloat = false;
                if (c == '-') i++;
                while (i < json.length() && (Character.isDigit(json.charAt(i)) || json.charAt(i) == '.')) {
                    if (json.charAt(i) == '.') isFloat = true;
                    i++;
                }
                String numStr = json.substring(start, i);
                map.put(key, isFloat ? Double.parseDouble(numStr) : Integer.parseInt(numStr));
            } else if (json.startsWith("true", i)) {
                map.put(key, true);
                i += 4;
            } else if (json.startsWith("false", i)) {
                map.put(key, false);
                i += 5;
            } else if (json.startsWith("null", i)) {
                map.put(key, null);
                i += 4;
            } else {
                throw new IllegalArgumentException("Unexpected character '" + c + "' at position " + i);
            }
        }
        return map;
    }

    /** Parse a JSON string starting at position i (which should be a '"'). Returns the raw content (without quotes). */
    private static String parseJsonString(String json, int i) {
        if (json.charAt(i) != '"') throw new IllegalArgumentException("Expected '\"'");
        StringBuilder sb = new StringBuilder();
        i++;
        while (i < json.length()) {
            char c = json.charAt(i);
            if (c == '\\') {
                i++;
                if (i >= json.length()) throw new IllegalArgumentException("Unexpected end in string escape");
                sb.append('\\').append(json.charAt(i));
                i++;
            } else if (c == '"') {
                return sb.toString();
            } else {
                sb.append(c);
                i++;
            }
        }
        throw new IllegalArgumentException("Unterminated string");
    }

    private static String unescapeJsonString(String s) {
        if (s == null || !s.contains("\\")) return s;
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char next = s.charAt(++i);
                switch (next) {
                    case '"'  -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case 'n'  -> sb.append('\n');
                    case 'r'  -> sb.append('\r');
                    case 't'  -> sb.append('\t');
                    case 'b'  -> sb.append('\b');
                    case 'f'  -> sb.append('\f');
                    case 'u'  -> {
                        if (i + 4 < s.length()) {
                            sb.append((char) Integer.parseInt(s.substring(i + 1, i + 5), 16));
                            i += 4;
                        }
                    }
                    default -> { sb.append('\\'); sb.append(next); }
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Utility
    // ══════════════════════════════════════════════════════════════════════

    @SuppressWarnings("unchecked")
    private static <K, V> Map<K, V> mapOf(Object... kvs) {
        Map<Object, Object> map = new LinkedHashMap<>();
        for (int i = 0; i < kvs.length; i += 2) {
            map.put(kvs[i], kvs[i + 1]);
        }
        return (Map<K, V>) map;
    }

    private static String stringVal(Map<String, Object> map, String key) {
        Object v = map.get(key);
        return v instanceof String s ? s : null;
    }

    private static int intVal(Map<String, Object> map, String key, int defaultVal) {
        Object v = map.get(key);
        if (v instanceof Number n) return n.intValue();
        if (v instanceof String s) {
            try { return Integer.parseInt(s); } catch (NumberFormatException e) { /* ignore */ }
        }
        return defaultVal;
    }

    private static double round2(double val) {
        return Math.round(val * 100.0) / 100.0;
    }
}

