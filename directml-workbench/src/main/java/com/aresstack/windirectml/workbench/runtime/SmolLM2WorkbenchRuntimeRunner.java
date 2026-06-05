package com.aresstack.windirectml.workbench.runtime;

import com.aresstack.windirectml.inference.smollm2.SmolLM2CompileOptions;
import com.aresstack.windirectml.inference.smollm2.SmolLM2GenerationOptions;
import com.aresstack.windirectml.inference.smollm2.SmolLM2GenerationDiagnostics;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Runtime;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimePackage;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeRequest;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeResult;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Tokenizer;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WdmlPackCompiler;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * Workbench adapter for the SmolLM2 reference runtime.
 *
 * <p>This class keeps the Swing panel free from package-loading and first-use
 * compile details. It deliberately targets the reference runtime only. The
 * optimized WARP path can replace this adapter later without changing the UI
 * dispatch.</p>
 */
public final class SmolLM2WorkbenchRuntimeRunner {

    private static final String DEFAULT_PACKAGE_FILE = "model.wdmlpack";
    private static final String TOKENIZER_FILE = "tokenizer.json";

    private final Path modelDir;
    private final SmolLM2WdmlPackCompiler compiler;

    public SmolLM2WorkbenchRuntimeRunner(Path modelDir) {
        this(modelDir, new SmolLM2WdmlPackCompiler());
    }

    SmolLM2WorkbenchRuntimeRunner(Path modelDir, SmolLM2WdmlPackCompiler compiler) {
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        this.compiler = Objects.requireNonNull(compiler, "compiler");
    }

    public Result generate(String prompt, int maxTokens) throws IOException {
        Path packagePath = ensureExecutablePackage();
        Path tokenizerPath = requireTokenizer();

        SmolLM2RuntimePackage runtimePackage = SmolLM2RuntimePackage.open(packagePath);
        SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerPath);
        try (SmolLM2Runtime runtime = SmolLM2Runtime.load(runtimePackage, tokenizer)) {
            SmolLM2RuntimeResult result = runtime.generate(new SmolLM2RuntimeRequest(
                    prompt,
                    maxTokens,
                    SmolLM2GenerationOptions.greedy()));
            return new Result(
                    result.generatedText(),
                    result.tokensGenerated(),
                    result.finishReason(),
                    runtimePackage.modelPackage().runtimeLoadMode(),
                    packagePath,
                    result.diagnostics());
        }
    }

    private Path ensureExecutablePackage() throws IOException {
        Path packagePath = modelDir.resolve(DEFAULT_PACKAGE_FILE);
        if (isExecutablePackage(packagePath)) {
            return packagePath;
        }
        if (!hasCompileSource()) {
            if (Files.isRegularFile(packagePath)) {
                throw new IllegalStateException("Existing SmolLM2 package is not executable: " + packagePath
                        + ". Put dense SafeTensors files into the model folder and retry so the package can be rebuilt.");
            }
            throw new IllegalStateException("Missing SmolLM2 runtime package and no SafeTensors source is available. "
                    + "Download the SmolLM2 model first from the Download tab.");
        }
        compiler.compile(new SmolLM2CompileOptions(modelDir, packagePath, false, true));
        if (!isExecutablePackage(packagePath)) {
            throw new IllegalStateException("Compiled SmolLM2 package is not executable: " + packagePath);
        }
        return packagePath;
    }

    private boolean isExecutablePackage(Path packagePath) {
        if (!Files.isRegularFile(packagePath)) {
            return false;
        }
        try {
            return SmolLM2RuntimePackage.open(packagePath).executable();
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean hasCompileSource() throws IOException {
        if (!Files.isRegularFile(modelDir.resolve("config.json"))) {
            return false;
        }
        if (!Files.isDirectory(modelDir)) {
            return false;
        }
        try (Stream<Path> stream = Files.list(modelDir)) {
            return stream
                    .filter(Files::isRegularFile)
                    .anyMatch(path -> path.getFileName().toString().endsWith(".safetensors"));
        }
    }

    private Path requireTokenizer() {
        Path tokenizer = modelDir.resolve(TOKENIZER_FILE);
        if (!Files.isRegularFile(tokenizer)) {
            throw new IllegalStateException("Missing SmolLM2 tokenizer.json: " + tokenizer
                    + ". Download the model first from the Download tab.");
        }
        return tokenizer;
    }

    public record Result(String text,
                         int outputTokens,
                         String finishReason,
                         String runtimeMode,
                         Path packagePath,
                         SmolLM2GenerationDiagnostics diagnostics) {
        public Result {
            text = text == null ? "" : text;
            finishReason = finishReason == null ? "" : finishReason;
            runtimeMode = runtimeMode == null || runtimeMode.isBlank() ? "reference" : runtimeMode;
            packagePath = Objects.requireNonNull(packagePath, "packagePath");
            diagnostics = Objects.requireNonNull(diagnostics, "diagnostics");
        }
    }
}
