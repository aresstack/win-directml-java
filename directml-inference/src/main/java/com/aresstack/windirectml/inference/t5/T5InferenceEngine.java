package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
    private CodeT5Tokenizer tokenizer;
    private T5RuntimePackage runtimePackage;
    private T5Runtime runtime;
    private boolean ready;

    public T5InferenceEngine(Path modelDir, int maxTokens) {
        this(modelDir, maxTokens, DEFAULT_MAX_INPUT_TOKENS);
    }

    public T5InferenceEngine(Path modelDir, int maxTokens, int maxInputTokens) {
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens must be positive: " + maxTokens);
        }
        if (maxInputTokens <= 0) {
            throw new IllegalArgumentException("maxInputTokens must be positive: " + maxInputTokens);
        }
        this.maxTokens = maxTokens;
        this.maxInputTokens = maxInputTokens;
    }

    @Override
    public void initialize() throws InferenceException {
        if (ready) {
            return;
        }
        try {
            validateTokenizerFiles(modelDir);
            tokenizer = CodeT5Tokenizer.load(modelDir);
            Path packagePath = resolveRuntimePackage(modelDir);
            runtimePackage = T5RuntimePackage.open(packagePath);
            runtime = T5Runtime.load(runtimePackage);
            ready = true;
        } catch (IOException | RuntimeException e) {
            throw new InferenceException("Could not initialize T5 runtime: " + e.getMessage(), e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        Objects.requireNonNull(request, "request");
        if (!ready) {
            throw new InferenceException("T5 runtime is not initialized");
        }
        try {
            String prompt = request.toFullPrompt();
            int[] inputTokens = truncate(tokenizer.encode(prompt), maxInputTokens);
            T5RuntimeRequest runtimeRequest = T5RuntimeRequest.greedy(inputTokens,
                    Math.min(request.getMaxTokens(), maxTokens), runtimePackage.metadata().specialTokens());
            long start = System.nanoTime();
            T5RuntimeResult result = runtime.generate(runtimeRequest);
            long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
            String text = tokenizer.decode(result.outputTokenIds());
            String finishReason = result.finishReason() == T5RuntimeResult.FinishReason.max_tokens
                    ? "max_tokens" : "end_turn";
            return new InferenceResult(text, finishReason,
                    new InferenceResult.Usage(inputTokens.length, result.generatedTokens(),
                            inputTokens.length + result.generatedTokens())) {
                @Override
                public String toString() {
                    return super.toString() + " elapsedMs=" + elapsedMs;
                }
            };
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
        runtimePackage = null;
        tokenizer = null;
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    public T5RuntimePackage runtimePackage() {
        return runtimePackage;
    }

    public static String describeMissingModelFile(Path modelDir) {
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            return "CodeT5 model directory not found: " + modelDir;
        }
        Path vocab = modelDir.resolve("vocab.json");
        if (!Files.isRegularFile(vocab)) {
            return "Missing CodeT5 tokenizer file: " + vocab.getFileName();
        }
        Path merges = modelDir.resolve("merges.txt");
        if (!Files.isRegularFile(merges)) {
            return "Missing CodeT5 tokenizer file: " + merges.getFileName();
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
        return "Missing T5 runtime package (*.wdmlpack) and no SafeTensors source is available for auto-compile";
    }

    private static Path resolveRuntimePackage(Path modelDir) throws IOException {
        Optional<Path> existing = findExistingWdmlPack(modelDir);
        if (existing.isPresent()) {
            return existing.get();
        }
        if (T5ModelImport.discoverSafeTensors(modelDir).isEmpty()) {
            throw new IOException("No T5 .wdmlpack found and no .safetensors source is available in " + modelDir);
        }
        T5CompileOptions options = new T5CompileOptions(modelDir, modelDir.resolve(DEFAULT_PACKAGE_NAME), false, true);
        return T5WdmlPackCompiler.compile(options).output();
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
        String missing = describeMissingModelFile(modelDir);
        if (missing != null) {
            throw new IOException(missing);
        }
    }

    private static int[] truncate(int[] values, int maxLength) {
        if (values.length <= maxLength) {
            return values;
        }
        return java.util.Arrays.copyOf(values, maxLength);
    }
}
