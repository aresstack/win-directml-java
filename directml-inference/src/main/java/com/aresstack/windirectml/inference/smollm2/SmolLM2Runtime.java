package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.GeneratedToken;
import com.aresstack.windirectml.inference.GenerationTokenSink;
import com.aresstack.windirectml.inference.prompt.PromptStrategies;
import com.aresstack.windirectml.inference.prompt.PromptStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Public SmolLM2 runtime facade.
 *
 * <p>The optimized WARP runtime is still a future work item. This facade exposes a correctness-first reference path
 * that can run token-level generation and, when a tokenizer is supplied, prompt-to-text generation.</p>
 */
public final class SmolLM2Runtime implements AutoCloseable {

    private static final String TOKENIZER_REQUIRED_MESSAGE =
            "SmolLM2 text generation requires a SmolLM2Tokenizer. Load the runtime with "
                    + "SmolLM2Runtime.load(runtimePackage, tokenizer) or call generateTokenIds(...).";

    /** Default ChatML prompt strategy for the SmolLM2 family (single source of truth). */
    private static final PromptStrategy DEFAULT_PROMPT_STRATEGY = PromptStrategies.forModel("smollm2");

    private final SmolLM2RuntimePackage runtimePackage;
    private final SmolLM2Tokenizer tokenizer;
    private final PromptStrategy promptStrategy;
    private final SmolLM2RuntimeMode runtimeMode;
    private final SmolLM2WarpRuntime warpRuntime;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private SmolLM2Runtime(SmolLM2RuntimePackage runtimePackage,
                           SmolLM2Tokenizer tokenizer,
                           SmolLM2RuntimeMode runtimeMode,
                           SmolLM2WarpRuntime warpRuntime,
                           PromptStrategy promptStrategy) {
        this.runtimePackage = Objects.requireNonNull(runtimePackage, "runtimePackage");
        this.tokenizer = tokenizer;
        this.runtimeMode = Objects.requireNonNull(runtimeMode, "runtimeMode");
        this.warpRuntime = warpRuntime;
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
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.REFERENCE, null, promptStrategy);
    }

    public static SmolLM2Runtime loadWarp(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength) {
        return loadWarp(runtimePackage, tokenizer, sequenceLength, SmolLM2WarpExecutorFactory.createDefaultExecutor());
    }

    public static SmolLM2Runtime loadWarp(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength,
                                          SmolLM2WarpExecutor executor) {
        SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(runtimePackage, sequenceLength, executor);
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.WARP, warpRuntime,
                DEFAULT_PROMPT_STRATEGY);
    }

    public static SmolLM2Runtime loadAuto(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength) {
        return loadAuto(runtimePackage, tokenizer, sequenceLength, SmolLM2WarpExecutorFactory.createDefaultExecutor());
    }

    public static SmolLM2Runtime loadAuto(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength,
                                          SmolLM2WarpExecutor executor) {
        SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(runtimePackage, sequenceLength, executor);
        SmolLM2RuntimeMode selectedMode = warpRuntime.executable() ? SmolLM2RuntimeMode.WARP : SmolLM2RuntimeMode.REFERENCE;
        if (selectedMode == SmolLM2RuntimeMode.REFERENCE) {
            warpRuntime.close();
            return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.REFERENCE, null,
                    DEFAULT_PROMPT_STRATEGY);
        }
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.WARP, warpRuntime,
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
            return warpRuntime.generateTokenIds(request);
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

    public Optional<SmolLM2WarpExecutionStatus> warpExecutionStatus() {
        return Optional.ofNullable(warpRuntime).map(SmolLM2WarpRuntime::status);
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true) && warpRuntime != null) {
            warpRuntime.close();
        }
    }
}
