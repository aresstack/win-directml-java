package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import com.aresstack.windirectml.inference.GeneratedToken;
import com.aresstack.windirectml.inference.GenerationTokenSink;
import com.aresstack.windirectml.inference.prompt.PromptInput;
import com.aresstack.windirectml.inference.prompt.PromptTask;
import com.aresstack.windirectml.inference.smollm2.SmolLM2GenerationOptions;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Runtime;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeRequest;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimeResult;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Package-backed {@link GenerationModelHandle} for SmolLM2, wrapping the native {@link SmolLM2Runtime}
 * (reference / WARP / AUTO) loaded from the SmolLM2 {@code model.wdmlpack}. Greedy chat decoding; the
 * runtime renders the ChatML prompt from the neutral {@link PromptInput}.
 */
final class SmolLm2GenerationHandle implements GenerationModelHandle {

    private final LocalRuntimeModelDescriptor descriptor;
    private final CatalogBackend backend;
    private final SmolLM2Runtime runtime;
    private volatile boolean closed;

    SmolLm2GenerationHandle(LocalRuntimeModelDescriptor descriptor, CatalogBackend backend,
            SmolLM2Runtime runtime) {
        this.descriptor = descriptor;
        this.backend = backend;
        this.runtime = runtime;
    }

    @Override
    public GenerationResult generate(GenerationRequest request) {
        return generate(request, null);
    }

    @Override
    public GenerationResult generate(GenerationRequest request, GenerationTokenListener listener) {
        if (request == null) {
            throw new GenerationException(GenerationErrorCode.INVALID_REQUEST, "request is null");
        }
        if (closed) {
            throw new GenerationException(GenerationErrorCode.GENERATION_FAILED,
                    "handle for " + runtimeModelId() + " is closed");
        }
        PromptInput promptInput = new PromptInput(PromptTask.NONE, request.userPrompt(),
                request.systemPrompt() == null ? "" : request.systemPrompt());
        SmolLM2RuntimeRequest runtimeRequest = new SmolLM2RuntimeRequest(
                promptInput, request.maxNewTokens(), SmolLM2GenerationOptions.greedyChat());
        try {
            SmolLM2RuntimeResult raw;
            if (listener == null) {
                raw = runtime.generate(runtimeRequest);
            } else {
                AtomicInteger index = new AtomicInteger();
                GenerationTokenSink sink = new GenerationTokenSink() {
                    @Override
                    public void onToken(GeneratedToken token) {
                        if (token == null) {
                            return;
                        }
                        String piece = token.delta();
                        if (piece != null && !piece.isEmpty()) {
                            listener.onToken(
                                    new GenerationToken(piece, token.tokenId(), index.getAndIncrement()));
                        }
                    }
                };
                raw = runtime.generate(runtimeRequest, sink);
            }
            int promptTokens = raw.diagnostics() == null ? 0 : raw.diagnostics().inputTokenCount();
            GenerationResult result = new GenerationResult(
                    raw.generatedText() == null ? "" : raw.generatedText(),
                    mapFinishReason(raw.finishReason()),
                    promptTokens,
                    raw.tokensGenerated(),
                    backend,
                    descriptor.runtimeModelId());
            if (listener != null) {
                listener.onComplete(result);
            }
            return result;
        } catch (RuntimeException e) {
            throw new GenerationException(GenerationErrorCode.GENERATION_FAILED,
                    "SmolLM2 generation failed for " + runtimeModelId() + ": " + e.getMessage(), e);
        }
    }

    private static GenerationFinishReason mapFinishReason(String reason) {
        String r = reason == null ? "" : reason.toLowerCase(Locale.ROOT);
        if (r.contains("length") || r.contains("max")) {
            return GenerationFinishReason.LENGTH;
        }
        return GenerationFinishReason.STOP;
    }

    @Override
    public CatalogBackend backend() {
        return backend;
    }

    @Override
    public String runtimeModelId() {
        return descriptor.runtimeModelId();
    }

    @Override
    public CatalogModelFamily family() {
        return descriptor.runtimeFamily();
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        runtime.close();
    }
}
