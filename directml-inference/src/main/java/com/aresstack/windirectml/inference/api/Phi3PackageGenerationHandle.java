package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import com.aresstack.windirectml.inference.phi3.Phi3Runtime;
import com.aresstack.windirectml.inference.phi3.Phi3Tokenizer;
import com.aresstack.windirectml.inference.phi3.Phi3Weights;
import com.aresstack.windirectml.inference.prompt.PromptInput;
import com.aresstack.windirectml.inference.prompt.PromptStrategies;
import com.aresstack.windirectml.inference.prompt.PromptTask;

/**
 * Package-backed {@link GenerationModelHandle} for Phi-3, mirroring the workbench's
 * {@code runPhi3Summarizer}: it loads exclusively from {@code model_phi3.wdmlpack} (via
 * {@code Phi3RuntimePackage}) and drives the native Java {@link Phi3Runtime} — no raw ONNX, no ONNX
 * Runtime, no Python. This CPU compute path is the demonstrated, package-only Phi-3 runtime.
 */
final class Phi3PackageGenerationHandle implements GenerationModelHandle {

    private final LocalRuntimeModelDescriptor descriptor;
    private final CatalogBackend backend;
    private final Phi3Weights weights;
    private final Phi3Tokenizer tokenizer;
    private final Phi3Runtime runtime;
    private volatile boolean closed;

    Phi3PackageGenerationHandle(LocalRuntimeModelDescriptor descriptor, CatalogBackend backend,
            Phi3Weights weights, Phi3Tokenizer tokenizer, Phi3Runtime runtime) {
        this.descriptor = descriptor;
        this.backend = backend;
        this.weights = weights;
        this.tokenizer = tokenizer;
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
        int[] index = {0};
        int promptTokenCount = tokenizer.encode(prompt).length;
        try {
            String text = runtime.generateStreaming(prompt, request.maxNewTokens(),
                    (tokenId, textSoFar, delta) -> {
                        index[0]++;
                        if (listener != null && delta != null && !delta.isEmpty()) {
                            listener.onToken(new GenerationToken(delta, tokenId, index[0] - 1));
                        }
                    });
            int generated = index[0];
            GenerationFinishReason finishReason = generated >= request.maxNewTokens()
                    ? GenerationFinishReason.LENGTH : GenerationFinishReason.STOP;
            GenerationResult result = new GenerationResult(
                    text == null ? "" : text.strip(), finishReason, promptTokenCount, generated,
                    backend, descriptor.runtimeModelId());
            if (listener != null) {
                listener.onComplete(result);
            }
            return result;
        } catch (RuntimeException e) {
            throw new GenerationException(GenerationErrorCode.GENERATION_FAILED,
                    "Phi-3 generation failed for " + runtimeModelId() + ": " + e.getMessage(), e);
        }
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
        try {
            weights.close();
        } catch (java.io.IOException | RuntimeException ignored) {
            // best-effort release of the mmap'd package weights
        }
    }
}
