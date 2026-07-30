package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import com.aresstack.windirectml.inference.gemma.Gemma3NativeWarpRuntime;
import com.aresstack.windirectml.inference.prompt.PromptInput;
import com.aresstack.windirectml.inference.prompt.PromptStrategies;
import com.aresstack.windirectml.inference.prompt.PromptTask;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Package-backed {@link GenerationModelHandle} for Gemma 3, wrapping {@link Gemma3NativeWarpRuntime}
 * (native Java/DirectML, WARP software adapter or AUTO hardware adapter). Loads from
 * {@code model_gemma3.wdmlpack}; never falls back to Python. The prompt is rendered through the Gemma
 * chat template here, so the runtime is called with {@code applyChatTemplate=false}.
 */
final class Gemma3GenerationHandle implements GenerationModelHandle {

    private final LocalRuntimeModelDescriptor descriptor;
    private final CatalogBackend backend;
    private final Gemma3NativeWarpRuntime runtime;
    private volatile boolean closed;

    Gemma3GenerationHandle(LocalRuntimeModelDescriptor descriptor, CatalogBackend backend,
            Gemma3NativeWarpRuntime runtime) {
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
        String prompt = PromptStrategies.forModel(descriptor.huggingFaceRepositoryId())
                .renderPrompt(new PromptInput(PromptTask.NONE, request.userPrompt(),
                        request.systemPrompt() == null ? "" : request.systemPrompt()));
        try {
            Gemma3NativeWarpRuntime.Result raw;
            if (listener == null) {
                raw = runtime.generate(prompt, false, request.maxNewTokens());
            } else {
                AtomicInteger index = new AtomicInteger();
                raw = runtime.generateStreaming(prompt, false, request.maxNewTokens(), delta -> {
                    if (delta != null && !delta.isEmpty()) {
                        listener.onToken(new GenerationToken(delta, -1, index.getAndIncrement()));
                    }
                });
            }
            GenerationResult result = new GenerationResult(
                    raw.text() == null ? "" : raw.text(),
                    mapFinishReason(raw.finishReason()),
                    raw.promptTokens(),
                    raw.outputTokens(),
                    backend,
                    descriptor.runtimeModelId());
            if (listener != null) {
                listener.onComplete(result);
            }
            return result;
        } catch (IOException | RuntimeException e) {
            throw new GenerationException(GenerationErrorCode.GENERATION_FAILED,
                    "Gemma 3 generation failed for " + runtimeModelId() + ": " + e.getMessage(), e);
        }
    }

    private static GenerationFinishReason mapFinishReason(Object finishReason) {
        String r = finishReason == null ? "" : finishReason.toString().toLowerCase(Locale.ROOT);
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
        // Gemma3NativeWarpRuntime opens and closes its DirectML device per generate() call, so the
        // handle holds no long-lived native resource. Mark closed to reject further generate() calls.
        closed = true;
    }
}
