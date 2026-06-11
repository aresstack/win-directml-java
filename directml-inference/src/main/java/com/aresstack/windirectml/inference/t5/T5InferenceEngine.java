package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.GeneratedToken;
import com.aresstack.windirectml.inference.GenerationTokenSink;
import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import com.aresstack.windirectml.windows.WindowsBindings;
import com.aresstack.windirectml.windows.WindowsNativeException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Workbench-facing inference engine for the curated CodeT5/T5 seq2seq family.
 *
 * <p>This engine intentionally consumes only an internal {@code .wdmlpack} at
 * runtime. If a SafeTensors source is available, it can compile the package on
 * first use. It does not execute ONNX, PyTorch, or Hugging Face formats through
 * a foreign runtime.</p>
 */
public final class T5InferenceEngine implements InferenceEngine {
    public static final String DEFAULT_PACKAGE_NAME = T5CompileOptions.DEFAULT_OUTPUT_NAME;
    private static final int DEFAULT_MAX_INPUT_TOKENS = 512;

    private final Path modelDir;
    private final int maxTokens;
    private final int maxInputTokens;
    private final String backend;
    private T5TextTokenizer tokenizer;
    private T5RuntimePackage runtimePackage;
    private T5Runtime runtime;
    private WindowsBindings windowsBindings;
    private T5GenerationMetrics lastGenerationMetrics = T5GenerationMetrics.empty();
    private String lastOutputTokenPreview = "[]";
    private boolean ready;

    public T5InferenceEngine(Path modelDir, int maxTokens) {
        this(modelDir, maxTokens, DEFAULT_MAX_INPUT_TOKENS, "reference");
    }

    public T5InferenceEngine(Path modelDir, int maxTokens, String backend) {
        this(modelDir, maxTokens, DEFAULT_MAX_INPUT_TOKENS, backend);
    }

    public T5InferenceEngine(Path modelDir, int maxTokens, int maxInputTokens) {
        this(modelDir, maxTokens, maxInputTokens, "reference");
    }

    public T5InferenceEngine(Path modelDir, int maxTokens, int maxInputTokens, String backend) {
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens must be positive: " + maxTokens);
        }
        if (maxInputTokens <= 0) {
            throw new IllegalArgumentException("maxInputTokens must be positive: " + maxInputTokens);
        }
        this.maxTokens = maxTokens;
        this.maxInputTokens = maxInputTokens;
        this.backend = normalizeBackend(backend);
    }

    @Override
    public void initialize() throws InferenceException {
        if (ready) {
            return;
        }
        try {
            validateTokenizerFiles(modelDir);
            tokenizer = T5TokenizerLoader.load(modelDir);
            Path packagePath = resolveRuntimePackage(modelDir);
            runtimePackage = T5RuntimePackage.open(packagePath);
            runtime = createRuntime(runtimePackage);
            ready = true;
        } catch (IOException | RuntimeException e) {
            closeWindowsBindingsQuietly();
            throw new InferenceException("Could not initialize T5 runtime: " + e.getMessage(), e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        return generate(request, null);
    }

    @Override
    public InferenceResult generate(InferenceRequest request, GenerationTokenSink sink) throws InferenceException {
        Objects.requireNonNull(request, "request");
        if (!ready) {
            throw new InferenceException("T5 runtime is not initialized");
        }
        try {
            String prompt = com.aresstack.windirectml.inference.prompt.PromptStrategies
                    .forModel(request.getModelId())
                    .renderPrompt(new com.aresstack.windirectml.inference.prompt.PromptInput(
                            request.getTask(), request.getUserPrompt(), request.getSystemPrompt()));
            long tokenizeStart = System.nanoTime();
            int[] inputTokens = truncate(tokenizer.encode(prompt), maxInputTokens);
            long tokenizationNanos = System.nanoTime() - tokenizeStart;
            T5RuntimeRequest runtimeRequest = T5RuntimeRequest.greedyText(inputTokens,
                    Math.min(request.getMaxTokens(), maxTokens), runtimePackage.metadata().specialTokens());
            long start = System.nanoTime();
            TokenDecodingSink decodingSink = sink == null ? null : new TokenDecodingSink(tokenizer, sink);
            T5RuntimeResult result = runtime.generate(runtimeRequest, decodingSink);
            long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
            long detokenizeStart = System.nanoTime();
            int[] outputTokenIds = result.outputTokenIds();
            String text = decodingSink == null ? tokenizer.decode(outputTokenIds) : decodingSink.textSoFar();
            lastOutputTokenPreview = tokenizer.describeTokens(outputTokenIds, 24);
            long detokenizationNanos = System.nanoTime() - detokenizeStart;
            lastGenerationMetrics = result.generationMetrics()
                    .withTextBoundaryTimings(tokenizationNanos, detokenizationNanos);
            String finishReason = result.finishReason() == T5RuntimeResult.FinishReason.max_tokens
                    ? "max_tokens" : "end_turn";
            InferenceResult inferenceResult = new InferenceResult(text, finishReason,
                    new InferenceResult.Usage(inputTokens.length, result.generatedTokens(),
                            inputTokens.length + result.generatedTokens())) {
                @Override
                public String toString() {
                    return super.toString() + " elapsedMs=" + elapsedMs;
                }
            };
            if (sink != null) {
                sink.onCompleted(inferenceResult);
            }
            return inferenceResult;
        } catch (RuntimeException e) {
            throw new InferenceException("T5 generation failed: " + e.getMessage(), e);
        }
    }


    @Override
    public void shutdown() {
        ready = false;
        if (runtime != null) {
            runtime.close();
            runtime = null;
        }
        closeWindowsBindingsQuietly();
        runtimePackage = null;
        tokenizer = null;
        lastOutputTokenPreview = "[]";
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    public T5RuntimePackage runtimePackage() {
        return runtimePackage;
    }

    public String backend() {
        return backend;
    }

    public String executionMode() {
        return runtime == null ? "not-initialized" : runtime.executionMode();
    }

    public T5GenerationMetrics lastGenerationMetrics() {
        return lastGenerationMetrics;
    }

    public String lastOutputTokenPreview() {
        return lastOutputTokenPreview;
    }

    public static String describeMissingModelFile(Path modelDir) {
        String tokenizerError = T5TokenizerLoader.describeMissingTokenizer(modelDir);
        if (tokenizerError != null) {
            return tokenizerError;
        }
        Path config = modelDir.resolve("config.json");
        if (!Files.isRegularFile(config)) {
            return "Missing T5 config file: " + config.getFileName();
        }
        if (findExistingWdmlPack(modelDir).isPresent()) {
            return null;
        }
        try {
            if (!T5ModelImport.discoverSafeTensors(modelDir).isEmpty()) {
                return null;
            }
        } catch (IOException ignored) {
            // Return the actionable package message below.
        }
        Path pytorchCheckpoint = modelDir.resolve("pytorch_model.bin");
        if (Files.isRegularFile(pytorchCheckpoint)) {
            return null;
        }
        return "Missing T5 runtime package (*.wdmlpack) and no supported import source is available for auto-compile. "
                + "Place model.safetensors, pytorch_model.bin, or a precompiled " + DEFAULT_PACKAGE_NAME
                + " in the T5 model directory.";
    }

    private static Path resolveRuntimePackage(Path modelDir) throws IOException {
        Optional<Path> existing = findExistingWdmlPack(modelDir);
        if (existing.isPresent()) {
            return existing.get();
        }
        boolean hasSafeTensors = !T5ModelImport.discoverSafeTensors(modelDir).isEmpty();
        boolean hasTorchCheckpoint = Files.isRegularFile(modelDir.resolve("pytorch_model.bin"));
        if (!hasSafeTensors && !hasTorchCheckpoint) {
            throw new IOException("No T5 .wdmlpack found and no supported import source is available in " + modelDir
                    + " (expected *.safetensors or pytorch_model.bin)");
        }
        T5CompileOptions options = new T5CompileOptions(modelDir, modelDir.resolve(DEFAULT_PACKAGE_NAME), false, true);
        return T5WdmlPackCompiler.compile(options).output();
    }

    private T5Runtime createRuntime(T5RuntimePackage packageToLoad) throws IOException {
        if ("reference".equals(backend)) {
            return T5Runtime.load(packageToLoad);
        }
        if (!WindowsBindings.isSupported()) {
            throw new IOException("T5 backend '" + backend + "' requires Windows WARP/AUTO DirectML bindings");
        }
        windowsBindings = new WindowsBindings();
        String nativeBackend = toNativeBackend(backend);
        try {
            windowsBindings.init(nativeBackend);
        } catch (WindowsNativeException e) {
            throw new IOException("Could not initialize T5 Windows backend '" + nativeBackend + "': "
                    + e.getMessage(), e);
        }
        if (!windowsBindings.hasDirectMl()) {
            throw new IOException("T5 backend '" + backend + "' did not provide a DirectML device");
        }
        return T5Runtime.loadWarp(packageToLoad, windowsBindings);
    }


    private static final class TokenDecodingSink implements GenerationTokenSink {
        private final T5TextTokenizer tokenizer;
        private final GenerationTokenSink delegate;
        private final List<Integer> tokenIds = new ArrayList<Integer>();
        private String textSoFar = "";

        private TokenDecodingSink(T5TextTokenizer tokenizer, GenerationTokenSink delegate) {
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
            this.delegate = Objects.requireNonNull(delegate, "delegate");
        }

        @Override
        public void onToken(GeneratedToken token) {
            if (token == null) {
                return;
            }
            if (token.tokenId() >= 0) {
                tokenIds.add(token.tokenId());
            }
            int[] ids = new int[tokenIds.size()];
            for (int i = 0; i < tokenIds.size(); i++) {
                ids[i] = tokenIds.get(i);
            }
            String decoded = tokenizer.decode(ids);
            String delta = decoded.startsWith(textSoFar)
                    ? decoded.substring(textSoFar.length())
                    : decoded;
            textSoFar = decoded;
            if (!delta.isEmpty()) {
                delegate.onToken(new GeneratedToken(token.tokenId(), textSoFar, delta));
            }
        }

        private String textSoFar() {
            return textSoFar;
        }
    }
    private void closeWindowsBindingsQuietly() {
        if (windowsBindings != null) {
            try {
                windowsBindings.close();
            } finally {
                windowsBindings = null;
            }
        }
    }

    private static String normalizeBackend(String value) {
        String normalized = value == null ? "reference" : value.trim().toLowerCase(java.util.Locale.ROOT);
        if (normalized.isEmpty()) {
            return "reference";
        }
        if ("cpu".equals(normalized)) {
            return "reference";
        }
        if ("dml".equals(normalized) || "directml".equals(normalized)) {
            return "auto";
        }
        if ("hybrid".equals(normalized)) {
            return "auto";
        }
        if ("reference".equals(normalized) || "warp".equals(normalized) || "auto".equals(normalized)) {
            return normalized;
        }
        throw new IllegalArgumentException("Unsupported T5 backend: " + value
                + " (expected reference, warp, or auto)");
    }

    private static String toNativeBackend(String backend) {
        return "warp".equals(backend) ? "warp" : "auto";
    }

    private static Optional<Path> findExistingWdmlPack(Path modelDir) {
        Path defaultPackage = modelDir.resolve(DEFAULT_PACKAGE_NAME).toAbsolutePath().normalize();
        if (Files.isRegularFile(defaultPackage)) {
            return Optional.of(defaultPackage);
        }
        try (var stream = Files.list(modelDir)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(".wdmlpack"))
                    .map(Path::toAbsolutePath)
                    .map(Path::normalize)
                    .sorted()
                    .findFirst();
        } catch (IOException e) {
            return Optional.empty();
        }
    }

    private static void validateTokenizerFiles(Path modelDir) throws IOException {
        String tokenizerError = T5TokenizerLoader.describeMissingTokenizer(modelDir);
        if (tokenizerError != null) {
            throw new IOException(tokenizerError);
        }
    }

    private static int[] truncate(int[] values, int maxLength) {
        if (values.length <= maxLength) {
            return values;
        }
        return java.util.Arrays.copyOf(values, maxLength);
    }
}
