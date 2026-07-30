package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import com.aresstack.windirectml.inference.GeneratedToken;
import com.aresstack.windirectml.inference.GenerationTokenSink;
import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.InferenceException;
import com.aresstack.windirectml.inference.InferenceRequest;
import com.aresstack.windirectml.inference.InferenceResult;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * {@link GenerationModelHandle} backed by any {@link InferenceEngine} implementation (Qwen, T5,
 * Phi-3). Maps the neutral {@link GenerationRequest}/{@link GenerationResult} contract onto the
 * engine's {@link InferenceRequest}/{@link InferenceResult}, using greedy/deterministic decoding
 * (temperature 0). Prompt templating is the engine's responsibility (its {@code PromptStrategy}).
 */
final class InferenceEngineGenerationHandle implements GenerationModelHandle {

    private final InferenceEngine engine;
    private final LocalRuntimeModelDescriptor descriptor;
    private final CatalogBackend backend;
    private volatile boolean closed;

    InferenceEngineGenerationHandle(
            InferenceEngine engine, LocalRuntimeModelDescriptor descriptor, CatalogBackend backend) {
        this.engine = engine;
        this.descriptor = descriptor;
        this.backend = backend;
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
        InferenceRequest inferenceRequest = InferenceRequest.builder()
                // PromptStrategies.forModel keys the chat-template/task strategy off the model id by
                // substring (e.g. "codet5-base-multi-sum", "qwen2.5-coder", "t5-small"), which the HF
                // repository id matches exactly — the underscore-cased runtimeModelId would not.
                .modelId(descriptor.huggingFaceRepositoryId())
                .userPrompt(request.userPrompt())
                .systemPrompt(request.systemPrompt() == null ? "" : request.systemPrompt())
                .maxTokens(request.maxNewTokens())
                .temperature(0.0f) // greedy / deterministic
                .build();
        try {
            InferenceResult raw;
            if (listener == null) {
                raw = engine.generate(inferenceRequest);
            } else {
                AtomicInteger index = new AtomicInteger();
                GenerationTokenSink sink = new GenerationTokenSink() {
                    @Override
                    public void onToken(GeneratedToken token) {
                        if (token == null) {
                            return;
                        }
                        String piece = token.delta();
                        if (piece == null || piece.isEmpty()) {
                            return;
                        }
                        listener.onToken(
                                new GenerationToken(piece, token.tokenId(), index.getAndIncrement()));
                    }
                };
                raw = engine.generate(inferenceRequest, sink);
            }
            GenerationResult result = toResult(raw);
            if (listener != null) {
                listener.onComplete(result);
            }
            return result;
        } catch (InferenceException e) {
            throw new GenerationException(GenerationErrorCode.GENERATION_FAILED,
                    "generation failed for " + runtimeModelId() + ": " + e.getMessage(), e);
        }
    }

    private GenerationResult toResult(InferenceResult raw) {
        String text = raw == null ? "" : raw.getText();
        int promptTokens = 0;
        int completionTokens = 0;
        if (raw != null && raw.getUsage() != null) {
            promptTokens = raw.getUsage().promptTokens();
            completionTokens = raw.getUsage().completionTokens();
        }
        GenerationFinishReason finishReason = mapFinishReason(raw == null ? null : raw.getFinishReason());
        return new GenerationResult(
                text, finishReason, promptTokens, completionTokens, backend, descriptor.runtimeModelId());
    }

    private static GenerationFinishReason mapFinishReason(String engineReason) {
        if ("max_tokens".equals(engineReason)) {
            return GenerationFinishReason.LENGTH;
        }
        if ("error".equals(engineReason)) {
            return GenerationFinishReason.ERROR;
        }
        // "end_turn" and any natural stop map to STOP.
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
        engine.shutdown();
    }
}
