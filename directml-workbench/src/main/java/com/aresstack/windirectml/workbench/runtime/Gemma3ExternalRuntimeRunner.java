package com.aresstack.windirectml.workbench.runtime;

import com.aresstack.windirectml.inference.prompt.PromptInput;
import com.aresstack.windirectml.inference.prompt.PromptStrategies;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/**
 * External Gemma 3 workbench runner.
 *
 * <p>This is intentionally not the native IronMind runtime. It is a narrowly scoped
 * bridge that lets the Workbench exercise a downloaded Gemma 3 directory through
 * local Python/Transformers while the real Java/WARP Gemma family is still absent.</p>
 */
public final class Gemma3ExternalRuntimeRunner {

    private static final String SCRIPT_RESOURCE = "/com/aresstack/windirectml/workbench/runtime/gemma3_generate.py";
    private static final long TIMEOUT_SECONDS = Long.getLong("gemma3.external.timeoutSeconds", 600L);

    private final Path modelDir;
    private final PythonCommand pythonCommand;

    public Gemma3ExternalRuntimeRunner(Path modelDir) {
        this(modelDir, PythonCommand.fromEnvironment());
    }

    Gemma3ExternalRuntimeRunner(Path modelDir, PythonCommand pythonCommand) {
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir");
        this.pythonCommand = Objects.requireNonNull(pythonCommand, "pythonCommand");
    }

    public Result generate(PromptInput input, int maxTokens) throws IOException, InterruptedException {
        validateModelFiles();
        if (maxTokens < 1) {
            throw new IllegalArgumentException("maxTokens must be positive");
        }
        Path tempDir = Files.createTempDirectory("gemma3-workbench-");
        try {
            Path script = extractScript(tempDir);
            Path promptFile = tempDir.resolve("prompt.txt");
            Path outputFile = tempDir.resolve("output.txt");
            String prompt = PromptStrategies.forModel("google/gemma-3-270m-it").renderPrompt(input);
            Files.writeString(promptFile, prompt, StandardCharsets.UTF_8);

            List<String> command = new ArrayList<String>();
            command.addAll(pythonCommand.parts());
            command.add(script.toString());
            command.add("--model-dir");
            command.add(modelDir.toString());
            command.add("--prompt-file");
            command.add(promptFile.toString());
            command.add("--out");
            command.add(outputFile.toString());
            command.add("--max-new-tokens");
            command.add(String.valueOf(maxTokens));

            ProcessBuilder builder = new ProcessBuilder(command);
            builder.environment().put("PYTHONIOENCODING", "utf-8");
            builder.environment().put("HF_HUB_OFFLINE", "1");
            Process process = builder.start();
            StreamCapture stdout = StreamCapture.start(process.getInputStream());
            StreamCapture stderr = StreamCapture.start(process.getErrorStream());
            boolean finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                process.waitFor(5, TimeUnit.SECONDS);
            }
            String stdoutText = stdout.await();
            String stderrText = stderr.await();
            if (!finished) {
                throw new IllegalStateException("Gemma 3 external probe timed out after "
                        + TIMEOUT_SECONDS + " seconds.");
            }
            if (process.exitValue() != 0) {
                throw new IllegalStateException("Gemma 3 external probe failed with exit code "
                        + process.exitValue() + ": " + explainFailure(stderrText));
            }
            String text = Files.exists(outputFile) ? Files.readString(outputFile, StandardCharsets.UTF_8) : "";
            Metrics metrics = Metrics.parse(stdoutText);
            return new Result(text, metrics, stdoutText, stderrText, pythonCommand.displayName(), modelDir);
        } finally {
            deleteRecursive(tempDir);
        }
    }

    public static String describeMissingModelFile(Path modelDir) {
        if (!Files.isDirectory(modelDir)) {
            return "Gemma 3 model directory is missing: " + modelDir;
        }
        String[] requiredFiles = {"model.safetensors", "config.json", "tokenizer.json"};
        for (String requiredFile : requiredFiles) {
            if (!Files.isRegularFile(modelDir.resolve(requiredFile))) {
                return "Missing Gemma 3 file: " + requiredFile;
            }
        }
        return null;
    }

    private void validateModelFiles() {
        String missing = describeMissingModelFile(modelDir);
        if (missing != null) {
            throw new IllegalStateException(missing + ". Download Gemma 3 from the Download tab first.");
        }
    }

    private static Path extractScript(Path tempDir) throws IOException {
        Path scriptPath = tempDir.resolve("gemma3_generate.py");
        try (InputStream input = Gemma3ExternalRuntimeRunner.class.getResourceAsStream(SCRIPT_RESOURCE)) {
            if (input == null) {
                throw new IllegalStateException("Missing workbench resource: " + SCRIPT_RESOURCE);
            }
            Files.copy(input, scriptPath);
        }
        return scriptPath;
    }

    private static String explainFailure(String stderrText) {
        String message = stderrText == null ? "" : stderrText.trim();
        if (message.isEmpty()) {
            return "no error output";
        }
        if (message.contains("Missing Python dependency")) {
            return message + " Use the same Python environment that has torch and transformers installed.";
        }
        if (message.contains("not currently available locally") || message.contains("Cannot find")) {
            return message + " The Workbench runs this Gemma path offline from the downloaded model directory.";
        }
        return message;
    }

    private static void deleteRecursive(Path path) {
        if (path == null || !Files.exists(path)) {
            return;
        }
        try {
            Files.walk(path)
                    .sorted((a, b) -> b.getNameCount() - a.getNameCount())
                    .forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                        } catch (IOException ignored) {
                            // Best-effort cleanup only.
                        }
                    });
        } catch (IOException ignored) {
            // Best-effort cleanup only.
        }
    }

    static final class PythonCommand {
        private final List<String> parts;

        private PythonCommand(List<String> parts) {
            if (parts.isEmpty()) {
                throw new IllegalArgumentException("python command must not be empty");
            }
            this.parts = List.copyOf(parts);
        }

        static PythonCommand fromEnvironment() {
            String command = System.getProperty("gemma3.python");
            if (command == null || command.isBlank()) {
                command = System.getenv("GEMMA3_PYTHON");
            }
            if (command == null || command.isBlank()) {
                command = "python";
            }
            return parse(command);
        }

        static PythonCommand parse(String command) {
            String[] rawParts = command.trim().split("\\s+");
            List<String> parts = new ArrayList<String>();
            for (String rawPart : rawParts) {
                if (!rawPart.isBlank()) {
                    parts.add(rawPart);
                }
            }
            return new PythonCommand(parts);
        }

        List<String> parts() {
            return parts;
        }

        String displayName() {
            return String.join(" ", parts);
        }
    }

    public static final class Result {
        private final String text;
        private final Metrics metrics;
        private final String stdout;
        private final String stderr;
        private final String pythonCommand;
        private final Path modelDir;

        Result(String text, Metrics metrics, String stdout, String stderr, String pythonCommand, Path modelDir) {
            this.text = text == null ? "" : text;
            this.metrics = Objects.requireNonNull(metrics, "metrics");
            this.stdout = stdout == null ? "" : stdout;
            this.stderr = stderr == null ? "" : stderr;
            this.pythonCommand = pythonCommand == null ? "" : pythonCommand;
            this.modelDir = Objects.requireNonNull(modelDir, "modelDir");
        }

        public String text() {
            return text;
        }

        public Metrics metrics() {
            return metrics;
        }

        public String stdout() {
            return stdout;
        }

        public String stderr() {
            return stderr;
        }

        public String pythonCommand() {
            return pythonCommand;
        }

        public Path modelDir() {
            return modelDir;
        }
    }

    public static final class Metrics {
        private final Map<String, Long> values;

        private Metrics(Map<String, Long> values) {
            this.values = Map.copyOf(values);
        }

        static Metrics parse(String stdout) {
            Map<String, Long> values = new LinkedHashMap<String, Long>();
            if (stdout != null) {
                for (String line : stdout.split("\\R")) {
                    int idx = line.indexOf('=');
                    if (idx <= 0) {
                        continue;
                    }
                    String key = line.substring(0, idx).trim();
                    String value = line.substring(idx + 1).trim();
                    if (key.startsWith("GEMMA3_")) {
                        try {
                            values.put(key, Long.parseLong(value));
                        } catch (NumberFormatException ignored) {
                            // Ignore non-numeric diagnostic lines.
                        }
                    }
                }
            }
            return new Metrics(values);
        }

        public long modelLoadMillis() {
            return value("GEMMA3_MODEL_LOAD_MS");
        }

        public long generateMillis() {
            return value("GEMMA3_GENERATE_MS");
        }

        public long promptTokens() {
            return value("GEMMA3_PROMPT_TOKENS");
        }

        public long outputTokens() {
            return value("GEMMA3_OUTPUT_TOKENS");
        }

        private long value(String key) {
            Long value = values.get(key);
            return value == null ? -1L : value;
        }
    }

    private static final class StreamCapture {
        private final Thread thread;
        private final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        private volatile IOException error;

        private StreamCapture(InputStream inputStream) {
            this.thread = new Thread(() -> capture(inputStream), "gemma3-stream-capture");
            this.thread.setDaemon(true);
        }

        static StreamCapture start(InputStream inputStream) {
            StreamCapture capture = new StreamCapture(inputStream);
            capture.thread.start();
            return capture;
        }

        String await() throws IOException, InterruptedException {
            thread.join();
            if (error != null) {
                throw error;
            }
            return buffer.toString(StandardCharsets.UTF_8);
        }

        private void capture(InputStream inputStream) {
            try (InputStream in = inputStream) {
                in.transferTo(buffer);
            } catch (IOException e) {
                error = e;
            }
        }
    }
}
