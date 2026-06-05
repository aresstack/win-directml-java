package com.aresstack.windirectml.workbench.runtime;

import com.aresstack.windirectml.inference.smollm2.SmolLM2ChatPromptTemplate;
import com.aresstack.windirectml.inference.smollm2.SmolLM2CompileOptions;
import com.aresstack.windirectml.inference.smollm2.SmolLM2GenerationOptions;
import com.aresstack.windirectml.inference.smollm2.SmolLM2GenerationDiagnostics;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Runtime;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeMode;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimePackage;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeRequest;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeResult;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WarpExecutionStatus;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WarpReadinessReport;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WarpRuntime;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Tokenizer;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WdmlPackCompiler;
import com.aresstack.windirectml.runtime.facade.Backend;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
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
        return generate(prompt, maxTokens, Backend.CPU);
    }

    public Result generate(String prompt, int maxTokens, Backend requestedBackend) throws IOException {
        Backend safeBackend = requestedBackend == null ? Backend.CPU : requestedBackend;
        Path packagePath = ensureExecutablePackage();
        Path tokenizerPath = requireTokenizer();

        SmolLM2RuntimePackage runtimePackage = SmolLM2RuntimePackage.open(packagePath);
        SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerPath);
        Optional<SmolLM2WarpReadinessReport> warpReadiness = inspectWarpReadinessIfRequested(runtimePackage, safeBackend);
        Optional<SmolLM2WarpExecutionStatus> warpStatus = warpReadiness.map(SmolLM2WorkbenchRuntimeRunner::toExecutionStatus);
        try (SmolLM2Runtime runtime = loadRuntime(runtimePackage, tokenizer, safeBackend, warpStatus)) {
            String summarizationPrompt = SmolLM2ChatPromptTemplate.defaultInstruct().renderSummarizationPrompt(prompt);
            SmolLM2RuntimeResult result = runtime.generate(new SmolLM2RuntimeRequest(
                    summarizationPrompt,
                    maxTokens,
                    SmolLM2GenerationOptions.greedy()));
            Optional<SmolLM2WarpExecutionStatus> effectiveWarpStatus = runtime.warpExecutionStatus().or(() -> warpStatus);
            return new Result(
                    cleanSummarizationOutput(result.generatedText()),
                    result.tokensGenerated(),
                    result.finishReason(),
                    safeBackend.name().toLowerCase(),
                    runtime.runtimeMode().name().toLowerCase(),
                    fallbackReason(safeBackend, runtime.runtimeMode(), effectiveWarpStatus),
                    warnings(effectiveWarpStatus),
                    packagePath,
                    result.diagnostics(),
                    warpReadiness);
        }
    }

    private static String cleanSummarizationOutput(String generatedText) {
        if (generatedText == null || generatedText.isBlank()) {
            return "";
        }

        String text = generatedText.replace("\r\n", "\n").replace('\r', '\n').trim();
        text = removeChatMarkers(text);
        text = removeInstructionEcho(text);
        text = removeRepeatedLeadLabels(text);
        return text.trim();
    }

    private static String removeChatMarkers(String text) {
        String cleaned = text;
        int assistantMarker = cleaned.lastIndexOf("<|im_start|>assistant");
        if (assistantMarker >= 0) {
            cleaned = cleaned.substring(assistantMarker + "<|im_start|>assistant".length());
        }
        return cleaned
                .replace("<|im_end|>", "")
                .replace("<|im_start|>", "")
                .trim();
    }

    private static String removeInstructionEcho(String text) {
        String cleaned = text.trim();
        String lower = cleaned.toLowerCase();
        if (lower.startsWith("fasse diesen quelltext")
                || lower.startsWith("fasse den folgenden quelltext")
                || lower.startsWith("aufgabe:")
                || lower.startsWith("regeln:")) {
            int summaryLabel = Math.max(cleaned.lastIndexOf("Zusammenfassung:"), cleaned.lastIndexOf("Fassung:"));
            if (summaryLabel >= 0) {
                int labelEnd = cleaned.indexOf(':', summaryLabel);
                cleaned = labelEnd >= 0 ? cleaned.substring(labelEnd + 1) : cleaned.substring(summaryLabel);
            }
        }
        return cleaned.trim();
    }

    private static String removeRepeatedLeadLabels(String text) {
        String cleaned = text.trim();
        boolean changed = true;
        while (changed) {
            changed = false;
            String next = removeLeadLabel(cleaned, "Zusammenfassung:");
            next = removeLeadLabel(next, "Fassung:");
            if (!next.equals(cleaned)) {
                cleaned = next.trim();
                changed = true;
            }
        }
        return cleaned;
    }

    private static String removeLeadLabel(String text, String label) {
        String cleaned = text.stripLeading();
        if (cleaned.startsWith(label)) {
            return cleaned.substring(label.length()).stripLeading();
        }
        return text;
    }

    private SmolLM2Runtime loadRuntime(SmolLM2RuntimePackage runtimePackage,
                                       SmolLM2Tokenizer tokenizer,
                                       Backend requestedBackend,
                                       Optional<SmolLM2WarpExecutionStatus> warpStatus) {
        if (requestedBackend == Backend.CPU) {
            return SmolLM2Runtime.loadReference(runtimePackage, tokenizer);
        }
        if (requestedBackend == Backend.AUTO && warpStatus.map(SmolLM2WarpExecutionStatus::executable).orElse(false)) {
            return SmolLM2Runtime.loadWarp(runtimePackage, tokenizer, runtimePackage.config().maxPositionEmbeddings());
        }
        if (isWarpLike(requestedBackend) && warpStatus.map(SmolLM2WarpExecutionStatus::executable).orElse(false)) {
            return SmolLM2Runtime.loadWarp(runtimePackage, tokenizer, runtimePackage.config().maxPositionEmbeddings());
        }
        return SmolLM2Runtime.loadReference(runtimePackage, tokenizer);
    }

    private Optional<SmolLM2WarpReadinessReport> inspectWarpReadinessIfRequested(SmolLM2RuntimePackage runtimePackage,
                                                                                   Backend requestedBackend) {
        if (requestedBackend == Backend.CPU) {
            return Optional.empty();
        }
        if (requestedBackend == Backend.AUTO || isWarpLike(requestedBackend)) {
            try (SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(
                    runtimePackage, runtimePackage.config().maxPositionEmbeddings())) {
                return Optional.of(SmolLM2WarpReadinessReport.fromRuntime(warpRuntime));
            }
        }
        return Optional.empty();
    }

    private static SmolLM2WarpExecutionStatus toExecutionStatus(SmolLM2WarpReadinessReport report) {
        return new SmolLM2WarpExecutionStatus(
                report.executable(),
                report.runtimeMode(),
                report.reason(),
                report.warnings());
    }

    private static boolean isWarpLike(Backend backend) {
        return backend == Backend.WARP || backend == Backend.DIRECTML || backend == Backend.HYBRID;
    }

    private static String fallbackReason(Backend requestedBackend,
                                         SmolLM2RuntimeMode runtimeMode,
                                         Optional<SmolLM2WarpExecutionStatus> warpStatus) {
        if (runtimeMode != SmolLM2RuntimeMode.REFERENCE || requestedBackend == Backend.CPU) {
            return "";
        }
        return warpStatus
                .filter(SmolLM2WarpExecutionStatus::requiresFallback)
                .map(status -> requestedBackend.name().toLowerCase()
                        + " requested, but SmolLM2 native WARP execution is not available yet: "
                        + status.reason())
                .orElse("");
    }

    private static List<String> warnings(Optional<SmolLM2WarpExecutionStatus> warpStatus) {
        return warpStatus.map(SmolLM2WarpExecutionStatus::warnings).orElseGet(List::of);
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
                         String requestedBackend,
                         String runtimeMode,
                         String fallbackReason,
                         List<String> runtimeWarnings,
                         Path packagePath,
                         SmolLM2GenerationDiagnostics diagnostics,
                         Optional<SmolLM2WarpReadinessReport> warpReadinessReport) {
        public Result {
            text = text == null ? "" : text;
            finishReason = finishReason == null ? "" : finishReason;
            requestedBackend = requestedBackend == null || requestedBackend.isBlank() ? "cpu" : requestedBackend;
            runtimeMode = runtimeMode == null || runtimeMode.isBlank() ? "reference" : runtimeMode;
            fallbackReason = fallbackReason == null ? "" : fallbackReason;
            runtimeWarnings = runtimeWarnings == null ? List.of() : List.copyOf(runtimeWarnings);
            packagePath = Objects.requireNonNull(packagePath, "packagePath");
            diagnostics = Objects.requireNonNull(diagnostics, "diagnostics");
            warpReadinessReport = warpReadinessReport == null ? Optional.empty() : warpReadinessReport;
        }
    }
}
