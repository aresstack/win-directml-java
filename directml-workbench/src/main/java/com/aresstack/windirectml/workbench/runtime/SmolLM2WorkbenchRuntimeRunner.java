package com.aresstack.windirectml.workbench.runtime;

import com.aresstack.windirectml.inference.GenerationTokenSink;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.artifact.ModelPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.SmolLM2PackageLifecycle;
import com.aresstack.windirectml.inference.prompt.PromptInput;
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
import com.aresstack.windirectml.inference.smollm2.SmolLM2ModelDirectory;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Tokenizer;
import com.aresstack.windirectml.runtime.facade.Backend;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Workbench adapter for the SmolLM2 reference runtime.
 *
 * <p>This class keeps the Swing panel free from package-loading and first-use
 * compile details. The CPU backend uses the reference runtime; the AUTO/WARP/DirectML
 * backends route to the native DirectML/WARP runtime when the WARP device is available,
 * and transparently fall back to the reference runtime otherwise.</p>
 */
public final class SmolLM2WorkbenchRuntimeRunner {

    private static final String TOKENIZER_FILE = "tokenizer.json";
    private static final String TOKENIZER_CONFIG_FILE = "tokenizer_config.json";

    private final Path modelDir;
    private final ModelPackageLifecycle lifecycle;

    public SmolLM2WorkbenchRuntimeRunner(Path modelDir) {
        this(modelDir, new SmolLM2PackageLifecycle());
    }

    SmolLM2WorkbenchRuntimeRunner(Path modelDir, ModelPackageLifecycle lifecycle) {
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
    }

    public Result generate(PromptInput promptInput, int maxTokens) throws IOException {
        return generate(promptInput, maxTokens, Backend.CPU, null);
    }

    public Result generate(PromptInput promptInput, int maxTokens, Backend requestedBackend) throws IOException {
        return generate(promptInput, maxTokens, requestedBackend, null);
    }

    public Result generate(PromptInput promptInput, int maxTokens, Backend requestedBackend,
                           GenerationTokenSink sink) throws IOException {
        PromptInput safeInput = promptInput == null ? PromptInput.raw("") : promptInput;
        Backend safeBackend = requestedBackend == null ? Backend.CPU : requestedBackend;

        Path packagePath = requireExecutablePackage();
        Path tokenizerPath = requireTokenizer();
        Path tokenizerConfigPath = optionalTokenizerConfig();

        SmolLM2RuntimePackage runtimePackage = SmolLM2RuntimePackage.open(packagePath);
        SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerPath, tokenizerConfigPath);
        Optional<SmolLM2WarpReadinessReport> warpReadiness = inspectWarpReadinessIfRequested(runtimePackage, safeBackend);
        Optional<SmolLM2WarpExecutionStatus> warpStatus = warpReadiness.map(SmolLM2WorkbenchRuntimeRunner::toExecutionStatus);
        try (SmolLM2Runtime runtime = loadRuntime(runtimePackage, tokenizer, safeBackend, warpStatus)) {
            SmolLM2RuntimeResult result = runtime.generate(new SmolLM2RuntimeRequest(
                    safeInput,
                    maxTokens,
                    SmolLM2GenerationOptions.greedyChat()), sink);
            Optional<SmolLM2WarpExecutionStatus> effectiveWarpStatus = runtime.warpExecutionStatus().or(() -> warpStatus);
            return new Result(
                    result.generatedText(),
                    result.tokensGenerated(),
                    result.finishReason(),
                    safeBackend.name().toLowerCase(),
                    runtime.runtimeModeDisplay(),
                    fallbackReason(safeBackend, runtime, effectiveWarpStatus),
                    warnings(effectiveWarpStatus),
                    packagePath,
                    result.diagnostics(),
                    warpReadiness,
                    effectiveConfigSummary(runtimePackage.config()));
        }
    }

    private static String effectiveConfigSummary(com.aresstack.windirectml.inference.smollm2.SmolLM2Config cfg) {
        return "hidden=" + cfg.hiddenSize()
                + ", layers=" + cfg.numHiddenLayers()
                + ", heads=" + cfg.numAttentionHeads()
                + ", kvHeads=" + cfg.effectiveKeyValueHeads()
                + ", headDim=" + cfg.effectiveHeadDim()
                + ", ropeTheta=" + cfg.ropeTheta()
                + ", rmsEps=" + cfg.rmsNormEps()
                + ", maxPos=" + cfg.maxPositionEmbeddings()
                + ", tieEmb=" + cfg.tieWordEmbeddings()
                + ", vocab=" + cfg.vocabSize()
                + ", bos=" + cfg.bosTokenId()
                + ", eos=" + cfg.eosTokenId();
    }

    private SmolLM2Runtime loadRuntime(SmolLM2RuntimePackage runtimePackage,
                                       SmolLM2Tokenizer tokenizer,
                                       Backend requestedBackend,
                                       Optional<SmolLM2WarpExecutionStatus> warpStatus) {
        if (requestedBackend == Backend.CPU) {
            return SmolLM2Runtime.loadReference(runtimePackage, tokenizer);
        }
        boolean warpReady = warpStatus.map(SmolLM2WarpExecutionStatus::executable).orElse(false);
        int maxPos = runtimePackage.config().maxPositionEmbeddings();
        if (requestedBackend == Backend.AUTO) {
            // AUTO uses a GPU if one exists (D3D12 hardware adapter), and falls back cleanly to the CPU reference
            // path when no usable device is available (including a lazy failure on first use).
            return warpReady
                    ? SmolLM2Runtime.loadAuto(runtimePackage, tokenizer, maxPos, "auto")
                    : SmolLM2Runtime.loadReference(runtimePackage, tokenizer);
        }
        // WARP is the default: the D3D12 WARP software rasterizer.
        return warpReady
                ? SmolLM2Runtime.loadWarp(runtimePackage, tokenizer, maxPos, "warp")
                : SmolLM2Runtime.loadReference(runtimePackage, tokenizer);
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
                                         SmolLM2Runtime runtime,
                                         Optional<SmolLM2WarpExecutionStatus> warpStatus) {
        if (runtime.runtimeMode() != SmolLM2RuntimeMode.REFERENCE || requestedBackend == Backend.CPU) {
            return "";
        }
        // A WARP device/upload failure that happened lazily on first use (AUTO mode) is the most precise reason.
        Optional<String> runtimeFallback = runtime.warpFallbackReason();
        if (runtimeFallback.isPresent()) {
            return requestedBackend.name().toLowerCase()
                    + " requested, but SmolLM2 native WARP execution failed at first use: "
                    + runtimeFallback.get();
        }
        return warpStatus
                .filter(SmolLM2WarpExecutionStatus::requiresFallback)
                .map(status -> requestedBackend.name().toLowerCase()
                        + " requested, but SmolLM2 native WARP execution is not available: "
                        + status.reason())
                .orElse("");
    }

    private static List<String> warnings(Optional<SmolLM2WarpExecutionStatus> warpStatus) {
        return warpStatus.map(SmolLM2WarpExecutionStatus::warnings).orElseGet(List::of);
    }

    /**
     * Resolve the runtime package to load, validating it through the central artifact lifecycle.
     * The runtime never compiles: if the package is missing/stale/corrupt/not-loadable/not-executable
     * this fails fast with an actionable error pointing at the manual Convert flow.
     */
    private Path requireExecutablePackage() {
        lifecycle.validateOrThrowBeforeInference(modelDir);
        return lifecycle.existingPackage(modelDir)
                .orElseThrow(() -> new IllegalStateException("Missing " + ModelFamily.SMOLLM2.displayName()
                        + " runtime package. Use Download tab -> Convert."));
    }

    private Path requireTokenizer() {
        Path tokenizer = modelDir.resolve(TOKENIZER_FILE);
        if (!Files.isRegularFile(tokenizer)) {
            throw new IllegalStateException("Missing SmolLM2 tokenizer.json: " + tokenizer
                    + ". Download the model first from the Download tab.");
        }
        return tokenizer;
    }

    private Path optionalTokenizerConfig() {
        Path tokenizerConfig = modelDir.resolve(TOKENIZER_CONFIG_FILE);
        if (Files.isRegularFile(tokenizerConfig)) {
            return tokenizerConfig;
        }
        return null;
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
                         Optional<SmolLM2WarpReadinessReport> warpReadinessReport,
                         String effectiveConfig) {
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
            effectiveConfig = effectiveConfig == null ? "" : effectiveConfig;
        }
    }
}
