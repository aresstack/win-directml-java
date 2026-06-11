package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.GeneratedToken;
import com.aresstack.windirectml.inference.GenerationTokenSink;
import com.aresstack.windirectml.inference.prompt.PromptStrategies;
import com.aresstack.windirectml.inference.prompt.PromptStrategy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Public SmolLM2 runtime facade.
 *
 * <p>Two execution paths are available: the correctness-first CPU reference path and the native DirectML/WARP path
 * ({@link SmolLM2NativeWarpExecutor}) which runs every dense projection on the shared decoder-only WARP kernels.
 * Use {@link #loadWarp} / {@link #loadAuto} for WARP execution and {@link #loadReference} for the CPU reference.</p>
 */
public final class SmolLM2Runtime implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(SmolLM2Runtime.class);

    /** When true ({@code -Dsmollm2.debug.prompt=true}) the rendered prompt, prompt token IDs
     *  and effective model config are logged for diagnostics. Off by default. */
    private static final boolean DEBUG_PROMPT = Boolean.getBoolean("smollm2.debug.prompt");

    private static final String TOKENIZER_REQUIRED_MESSAGE =
            "SmolLM2 text generation requires a SmolLM2Tokenizer. Load the runtime with "
                    + "SmolLM2Runtime.load(runtimePackage, tokenizer) or call generateTokenIds(...).";

    /** Default ChatML prompt strategy for the SmolLM2 family (single source of truth). */
    private static final PromptStrategy DEFAULT_PROMPT_STRATEGY = PromptStrategies.forModel("smollm2");

    private final SmolLM2RuntimePackage runtimePackage;
    private final SmolLM2Tokenizer tokenizer;
    private final PromptStrategy promptStrategy;
    private volatile SmolLM2RuntimeMode runtimeMode;
    private final SmolLM2WarpRuntime warpRuntime;
    private final boolean warpFallbackAllowed;
    /** The D3D12 adapter the native projection path runs on ("warp" software rasterizer or "directml" hardware). */
    private final String warpAdapterBackend;
    private volatile String warpFallbackReason = "";
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private SmolLM2Runtime(SmolLM2RuntimePackage runtimePackage,
                           SmolLM2Tokenizer tokenizer,
                           SmolLM2RuntimeMode runtimeMode,
                           SmolLM2WarpRuntime warpRuntime,
                           boolean warpFallbackAllowed,
                           PromptStrategy promptStrategy) {
        this(runtimePackage, tokenizer, runtimeMode, warpRuntime, warpFallbackAllowed, promptStrategy, "warp");
    }

    private SmolLM2Runtime(SmolLM2RuntimePackage runtimePackage,
                           SmolLM2Tokenizer tokenizer,
                           SmolLM2RuntimeMode runtimeMode,
                           SmolLM2WarpRuntime warpRuntime,
                           boolean warpFallbackAllowed,
                           PromptStrategy promptStrategy,
                           String warpAdapterBackend) {
        this.runtimePackage = Objects.requireNonNull(runtimePackage, "runtimePackage");
        this.tokenizer = tokenizer;
        this.runtimeMode = Objects.requireNonNull(runtimeMode, "runtimeMode");
        this.warpRuntime = warpRuntime;
        this.warpFallbackAllowed = warpFallbackAllowed;
        this.warpAdapterBackend = warpAdapterBackend == null || warpAdapterBackend.isBlank()
                ? "warp" : warpAdapterBackend.trim();
        this.promptStrategy = promptStrategy == null ? DEFAULT_PROMPT_STRATEGY : promptStrategy;
    }

    public static SmolLM2Runtime load(SmolLM2RuntimePackage runtimePackage) {
        return loadReference(runtimePackage, null);
    }

    public static SmolLM2Runtime load(SmolLM2RuntimePackage runtimePackage, SmolLM2Tokenizer tokenizer) {
        return loadReference(runtimePackage, Objects.requireNonNull(tokenizer, "tokenizer"));
    }

    public static SmolLM2Runtime loadReference(SmolLM2RuntimePackage runtimePackage, SmolLM2Tokenizer tokenizer) {
        return loadReference(runtimePackage, tokenizer, DEFAULT_PROMPT_STRATEGY);
    }

    public static SmolLM2Runtime loadReference(SmolLM2RuntimePackage runtimePackage,
                                               SmolLM2Tokenizer tokenizer,
                                               PromptStrategy promptStrategy) {
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.REFERENCE, null, false, promptStrategy);
    }

    public static SmolLM2Runtime loadWarp(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength) {
        return loadWarp(runtimePackage, tokenizer, sequenceLength, "warp");
    }

    /**
     * Load the native projection runtime bound to a specific D3D12 device family: {@code "warp"} for the software
     * rasterizer (the default) or {@code "auto"} for a hardware GPU when one exists. The kernels and numerics are
     * identical on both — only the device differs.
     */
    public static SmolLM2Runtime loadWarp(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength,
                                          String adapterBackend) {
        SmolLM2WarpExecutor executor = SmolLM2WarpExecutorFactory.createDefaultExecutor(adapterBackend);
        SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(runtimePackage, sequenceLength, executor);
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.WARP, warpRuntime, false,
                DEFAULT_PROMPT_STRATEGY, adapterBackend);
    }

    public static SmolLM2Runtime loadWarp(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength,
                                          SmolLM2WarpExecutor executor) {
        SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(runtimePackage, sequenceLength, executor);
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.WARP, warpRuntime, false,
                DEFAULT_PROMPT_STRATEGY);
    }

    public static SmolLM2Runtime loadAuto(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength) {
        return loadAuto(runtimePackage, tokenizer, sequenceLength, "warp");
    }

    /**
     * Load the native projection runtime in AUTO mode bound to a specific device family, falling back to the CPU
     * reference path when no usable device is available (including a lazy failure on first use).
     */
    public static SmolLM2Runtime loadAuto(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength,
                                          String adapterBackend) {
        SmolLM2WarpExecutor executor = SmolLM2WarpExecutorFactory.createDefaultExecutor(adapterBackend);
        SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(runtimePackage, sequenceLength, executor);
        SmolLM2RuntimeMode selectedMode = warpRuntime.executable() ? SmolLM2RuntimeMode.WARP : SmolLM2RuntimeMode.REFERENCE;
        if (selectedMode == SmolLM2RuntimeMode.REFERENCE) {
            warpRuntime.close();
            return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.REFERENCE, null, false,
                    DEFAULT_PROMPT_STRATEGY, adapterBackend);
        }
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.WARP, warpRuntime, true,
                DEFAULT_PROMPT_STRATEGY, adapterBackend);
    }

    public static SmolLM2Runtime loadAuto(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength,
                                          SmolLM2WarpExecutor executor) {
        SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(runtimePackage, sequenceLength, executor);
        SmolLM2RuntimeMode selectedMode = warpRuntime.executable() ? SmolLM2RuntimeMode.WARP : SmolLM2RuntimeMode.REFERENCE;
        if (selectedMode == SmolLM2RuntimeMode.REFERENCE) {
            warpRuntime.close();
            return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.REFERENCE, null, false,
                    DEFAULT_PROMPT_STRATEGY);
        }
        // AUTO: the WARP readiness probe is optimistic (it does not upload weights), so the real device/upload
        // initialisation happens lazily on the first generate call. Allow a clean fallback to the reference path
        // if that lazy initialisation fails, instead of surfacing the failure to the caller.
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.WARP, warpRuntime, true,
                DEFAULT_PROMPT_STRATEGY);
    }

    public SmolLM2RuntimeResult generate(SmolLM2RuntimeRequest request) {
        return generate(request, null);
    }

    public SmolLM2RuntimeResult generate(SmolLM2RuntimeRequest request, GenerationTokenSink sink) {
        Objects.requireNonNull(request, "request");
        long runtimeStart = System.nanoTime();
        ensureOpen();
        SmolLM2Tokenizer activeTokenizer = tokenizer().orElseThrow(
                () -> new SmolLM2RuntimeUnsupportedException(TOKENIZER_REQUIRED_MESSAGE));

        long tokenizeStart = System.nanoTime();
        String renderedPrompt = promptStrategy.renderPrompt(request.prompt());
        List<Integer> inputTokenIds = encodePrompt(renderedPrompt, activeTokenizer);
        long tokenizeNanos = System.nanoTime() - tokenizeStart;

        logPromptDiagnostics(request, renderedPrompt, inputTokenIds);

        StreamingTextBridge streamingTextBridge = sink == null
                ? null
                : new StreamingTextBridge(activeTokenizer, sink);
        SmolLM2TokenRuntimeResult tokenResult = generateTokenIds(new SmolLM2TokenRuntimeRequest(
                inputTokenIds, request.maxNewTokens(), request.options()),
                streamingTextBridge);

        long detokenizeStart = System.nanoTime();
        String generatedText = streamingTextBridge == null
                ? activeTokenizer.decode(toIntArray(tokenResult.generatedTokenIds()), true)
                : streamingTextBridge.textSoFar();
        long detokenizeNanos = System.nanoTime() - detokenizeStart;

        SmolLM2GenerationProfile profile = tokenResult.profile().withTextTimings(
                System.nanoTime() - runtimeStart,
                tokenizeNanos,
                detokenizeNanos);
        SmolLM2RuntimeResult result = new SmolLM2RuntimeResult(
                generatedText,
                tokenResult.generatedTokenIds(),
                tokenResult.tokensGenerated(),
                tokenResult.finishReason(),
                SmolLM2GenerationDiagnostics.fromTokenResult(tokenResult, generatedText, profile));
        if (sink != null) {
            sink.onCompleted(new com.aresstack.windirectml.inference.InferenceResult(
                    result.generatedText(),
                    result.finishReason(),
                    new com.aresstack.windirectml.inference.InferenceResult.Usage(
                            tokenResult.inputTokenCount(),
                            result.tokensGenerated(),
                            tokenResult.fullTokenCount())));
        }
        return result;
    }

    /**
     * Generate token IDs through the correctness-first reference pipeline.
     */
    public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request) {
        return generateTokenIds(request, null);
    }

    public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request,
                                                     java.util.function.IntConsumer generatedTokenConsumer) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        ensureExecutablePackage();
        if (runtimeMode == SmolLM2RuntimeMode.WARP) {
            try {
                return warpRuntime.generateTokenIds(request, generatedTokenConsumer);
            } catch (SmolLM2RuntimeUnsupportedException e) {
                if (!warpFallbackAllowed) {
                    throw e;
                }
                // AUTO mode: the lazy WARP device/upload initialisation failed on first use. Fall back to the
                // correctness-first reference path. The failure happens before any token is emitted to the
                // consumer (the engine is built up front in the session), so no token is streamed twice.
                this.warpFallbackReason = safeMessage(e);
                this.runtimeMode = SmolLM2RuntimeMode.REFERENCE;
                log.warn("SmolLM2 AUTO mode falling back from WARP to the reference runtime: {}", warpFallbackReason);
                warpRuntime.close();
            }
        }
        SmolLM2Weights weights = runtimePackage.requireWeights();
        return new SmolLM2ReferenceGenerationLoop(weights).generate(request, generatedTokenConsumer);
    }

    public Optional<SmolLM2Tokenizer> tokenizer() {
        return Optional.ofNullable(tokenizer);
    }

    private List<Integer> encodePrompt(String prompt, SmolLM2Tokenizer activeTokenizer) {
        int[] encoded = activeTokenizer.encode(prompt);
        List<Integer> inputTokenIds = new ArrayList<>(Math.max(1, encoded.length));
        for (int tokenId : encoded) {
            inputTokenIds.add(tokenId);
        }
        if (inputTokenIds.isEmpty()) {
            inputTokenIds.add(runtimePackage.config().bosTokenId());
        }
        return inputTokenIds;
    }

    /**
     * Log the actual prompt fed to the model: the rendered chat string and the
     * resulting token IDs. This is ground truth for diagnosing "model ignores the
     * task / produces base-continuation" reports — it reveals whether the ChatML
     * structure and the task instruction are present, and whether the special
     * tokens collapse to single IDs instead of literal byte text.
     */
    private void logPromptDiagnostics(SmolLM2RuntimeRequest request, String renderedPrompt, List<Integer> inputTokenIds) {
        if (!DEBUG_PROMPT || !log.isInfoEnabled()) {
            return;
        }
        SmolLM2Config cfg = runtimePackage.config();
        log.info("SmolLM2 effective config: hidden={}, layers={}, heads={}, kvHeads={}, headDim={}, "
                        + "ropeTheta={}, rmsEps={}, maxPos={}, tieEmb={}, vocab={}, bos={}, eos={}",
                cfg.hiddenSize(), cfg.numHiddenLayers(), cfg.numAttentionHeads(), cfg.effectiveKeyValueHeads(),
                cfg.effectiveHeadDim(), cfg.ropeTheta(), cfg.rmsNormEps(), cfg.maxPositionEmbeddings(),
                cfg.tieWordEmbeddings(), cfg.vocabSize(), cfg.bosTokenId(), cfg.eosTokenId());
        log.info("SmolLM2 prompt [task={}]: {} tokens; promptTokenIds={}; renderedPrompt=<<<{}>>>",
                request.prompt() == null ? null : request.prompt().task(),
                inputTokenIds.size(),
                inputTokenIds,
                renderedPrompt);
    }

    private static int[] toIntArray(List<Integer> tokenIds) {
        int[] ids = new int[tokenIds.size()];
        for (int i = 0; i < tokenIds.size(); i++) {
            ids[i] = tokenIds.get(i);
        }
        return ids;
    }


    private static final class StreamingTextBridge implements java.util.function.IntConsumer {
        private final SmolLM2Tokenizer tokenizer;
        private final GenerationTokenSink sink;
        private final List<Integer> tokenIds = new ArrayList<>();
        private String textSoFar = "";

        private StreamingTextBridge(SmolLM2Tokenizer tokenizer, GenerationTokenSink sink) {
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
            this.sink = Objects.requireNonNull(sink, "sink");
        }

        @Override
        public void accept(int tokenId) {
            tokenIds.add(tokenId);
            String decoded = tokenizer.decode(toIntArray(tokenIds), true);
            String delta = decoded.startsWith(textSoFar)
                    ? decoded.substring(textSoFar.length())
                    : decoded;
            textSoFar = decoded;
            if (!delta.isEmpty()) {
                sink.onToken(new GeneratedToken(tokenId, textSoFar, delta));
            }
        }

        private String textSoFar() {
            return textSoFar;
        }
    }
    private void ensureExecutablePackage() {
        if (!runtimePackage.executable()) {
            throw new SmolLM2RuntimeUnsupportedException(runtimePackage.runtimeLoadableReason());
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("SmolLM2 runtime is closed");
        }
    }

    public SmolLM2RuntimePackage runtimePackage() {
        return runtimePackage;
    }

    public SmolLM2RuntimeMode runtimeMode() {
        return runtimeMode;
    }

    /**
     * Honest, human-facing runtime-mode label that reflects how the native projection path actually runs. For the
     * reference path this is the CPU reference label; for the native path it names the selected device family: WARP
     * (the D3D12 software rasterizer, the default) or AUTO (a hardware GPU when one exists).
     */
    public String runtimeModeDisplay() {
        if (runtimeMode == SmolLM2RuntimeMode.REFERENCE) {
            return SmolLM2RuntimeMode.REFERENCE.displayLabel();
        }
        if ("auto".equalsIgnoreCase(warpAdapterBackend)) {
            return "auto (GPU projection path; norms/RoPE/attention/KV-cache on CPU)";
        }
        return "warp (WARP projection path; norms/RoPE/attention/KV-cache on CPU)";
    }

    public Optional<SmolLM2WarpExecutionStatus> warpExecutionStatus() {
        return Optional.ofNullable(warpRuntime).map(SmolLM2WarpRuntime::status);
    }

    /**
     * Reason the AUTO runtime fell back from WARP to the reference path at first use, if it did.
     * Empty when no runtime fallback occurred.
     */
    public Optional<String> warpFallbackReason() {
        return warpFallbackReason.isBlank() ? Optional.empty() : Optional.of(warpFallbackReason);
    }

    private static String safeMessage(Throwable e) {
        return e.getMessage() == null || e.getMessage().isBlank()
                ? e.getClass().getSimpleName()
                : e.getMessage();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true) && warpRuntime != null) {
            warpRuntime.close();
        }
    }
}
