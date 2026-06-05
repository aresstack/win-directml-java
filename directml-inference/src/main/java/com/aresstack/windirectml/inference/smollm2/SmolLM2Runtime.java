package com.aresstack.windirectml.inference.smollm2;

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

    private final SmolLM2RuntimePackage runtimePackage;
    private final SmolLM2Tokenizer tokenizer;
    private final SmolLM2ChatPromptTemplate chatPromptTemplate;
    private final SmolLM2RuntimeMode runtimeMode;
    private final SmolLM2WarpRuntime warpRuntime;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private SmolLM2Runtime(SmolLM2RuntimePackage runtimePackage,
                           SmolLM2Tokenizer tokenizer,
                           SmolLM2RuntimeMode runtimeMode,
                           SmolLM2WarpRuntime warpRuntime) {
        this.runtimePackage = Objects.requireNonNull(runtimePackage, "runtimePackage");
        this.tokenizer = tokenizer;
        this.runtimeMode = Objects.requireNonNull(runtimeMode, "runtimeMode");
        this.warpRuntime = warpRuntime;
        this.chatPromptTemplate = SmolLM2ChatPromptTemplate.defaultInstruct();
    }

    public static SmolLM2Runtime load(SmolLM2RuntimePackage runtimePackage) {
        return loadReference(runtimePackage, null);
    }

    public static SmolLM2Runtime load(SmolLM2RuntimePackage runtimePackage, SmolLM2Tokenizer tokenizer) {
        return loadReference(runtimePackage, Objects.requireNonNull(tokenizer, "tokenizer"));
    }

    public static SmolLM2Runtime loadReference(SmolLM2RuntimePackage runtimePackage, SmolLM2Tokenizer tokenizer) {
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.REFERENCE, null);
    }

    public static SmolLM2Runtime loadWarp(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength) {
        return loadWarp(runtimePackage, tokenizer, sequenceLength, new SmolLM2UnsupportedWarpExecutor());
    }

    public static SmolLM2Runtime loadWarp(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength,
                                          SmolLM2WarpExecutor executor) {
        SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(runtimePackage, sequenceLength, executor);
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.WARP, warpRuntime);
    }

    public static SmolLM2Runtime loadAuto(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength) {
        return loadAuto(runtimePackage, tokenizer, sequenceLength, new SmolLM2UnsupportedWarpExecutor());
    }

    public static SmolLM2Runtime loadAuto(SmolLM2RuntimePackage runtimePackage,
                                          SmolLM2Tokenizer tokenizer,
                                          int sequenceLength,
                                          SmolLM2WarpExecutor executor) {
        SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(runtimePackage, sequenceLength, executor);
        SmolLM2RuntimeMode selectedMode = warpRuntime.executable() ? SmolLM2RuntimeMode.WARP : SmolLM2RuntimeMode.REFERENCE;
        if (selectedMode == SmolLM2RuntimeMode.REFERENCE) {
            warpRuntime.close();
            return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.REFERENCE, null);
        }
        return new SmolLM2Runtime(runtimePackage, tokenizer, SmolLM2RuntimeMode.WARP, warpRuntime);
    }

    public SmolLM2RuntimeResult generate(SmolLM2RuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        SmolLM2Tokenizer activeTokenizer = tokenizer().orElseThrow(
                () -> new SmolLM2RuntimeUnsupportedException(TOKENIZER_REQUIRED_MESSAGE));
        String renderedPrompt = chatPromptTemplate.renderUserPrompt(request.prompt());
        List<Integer> inputTokenIds = encodePrompt(renderedPrompt, activeTokenizer);
        SmolLM2TokenRuntimeResult tokenResult = generateTokenIds(new SmolLM2TokenRuntimeRequest(
                inputTokenIds, request.maxNewTokens(), request.options()));
        String generatedText = activeTokenizer.decode(toIntArray(tokenResult.generatedTokenIds()), true);
        return new SmolLM2RuntimeResult(
                generatedText,
                tokenResult.generatedTokenIds(),
                tokenResult.tokensGenerated(),
                tokenResult.finishReason(),
                SmolLM2GenerationDiagnostics.fromTokenResult(tokenResult, generatedText));
    }

    /**
     * Generate token IDs through the correctness-first reference pipeline.
     */
    public SmolLM2TokenRuntimeResult generateTokenIds(SmolLM2TokenRuntimeRequest request) {
        Objects.requireNonNull(request, "request");
        ensureOpen();
        ensureExecutablePackage();
        if (runtimeMode == SmolLM2RuntimeMode.WARP) {
            return warpRuntime.generateTokenIds(request);
        }
        SmolLM2Weights weights = runtimePackage.requireWeights();
        return new SmolLM2ReferenceGenerationLoop(weights).generate(request);
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
