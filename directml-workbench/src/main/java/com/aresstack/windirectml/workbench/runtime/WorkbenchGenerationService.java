package com.aresstack.windirectml.workbench.runtime;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.LocalModelCatalog;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import com.aresstack.windirectml.inference.api.GenerationErrorCode;
import com.aresstack.windirectml.inference.api.GenerationException;
import com.aresstack.windirectml.inference.api.GenerationModelHandle;
import com.aresstack.windirectml.inference.api.GenerationRequest;
import com.aresstack.windirectml.inference.api.GenerationResult;
import com.aresstack.windirectml.inference.api.GenerationRuntime;
import com.aresstack.windirectml.inference.api.GenerationTokenListener;
import com.aresstack.windirectml.inference.api.LoadPolicy;
import com.aresstack.windirectml.inference.prompt.PromptInput;
import com.aresstack.windirectml.inference.prompt.PromptStrategies;
import com.aresstack.windirectml.inference.prompt.PromptTask;
import com.aresstack.windirectml.runtime.facade.Backend;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Swing-free bridge that runs generation through the shared, neutral
 * {@link GenerationRuntime} — the exact runtime AskAI's sidecar uses. The workbench UI keeps no
 * competing per-family orchestration: it resolves the selected model against the neutral
 * {@link LocalModelCatalog} (by Hugging Face repository id) and delegates to the runtime, which
 * enforces the backend matrix and loads package-only. No family engines, no Python bridge here.
 */
public final class WorkbenchGenerationService {

    /**
     * Run generation for a catalog model.
     *
     * @param modelRepositoryId the Hugging Face repository id (the workbench's selected model id)
     * @param modelDir          the on-disk model directory (resolved by the caller)
     * @param backend           the requested backend
     * @param promptTask        the selected task ({@code null} treated as {@link PromptTask#NONE})
     * @param userPrompt        the raw user text
     * @param maxNewTokens      generation cap
     * @param listener          optional per-token listener for streaming ({@code null} = buffered)
     * @return the neutral generation result
     * @throws GenerationException with a typed {@link GenerationErrorCode} on any failure
     */
    public GenerationResult generate(String modelRepositoryId, Path modelDir, CatalogBackend backend,
            PromptTask promptTask, String userPrompt, int maxNewTokens, GenerationTokenListener listener) {
        Objects.requireNonNull(modelDir, "modelDir");
        Objects.requireNonNull(backend, "backend");
        LocalRuntimeModelDescriptor descriptor = LocalModelCatalog.findByRepositoryId(modelRepositoryId);
        if (descriptor == null) {
            throw new GenerationException(GenerationErrorCode.INVALID_REQUEST,
                    "no catalog entry for model '" + modelRepositoryId + "'");
        }
        // Apply the selected task via the SAME family-aware PromptStrategies the runtime uses; this is not
        // workbench prompt logic, it delegates to the shared strategy. The runtime renders once more with
        // PromptTask.NONE, but every strategy is idempotent (it detects its own already-rendered markers /
        // task prefix and passes through), so the family template is applied exactly once.
        String renderedPrompt = renderPrompt(modelRepositoryId, promptTask, userPrompt);
        GenerationRequest request = GenerationRequest.builder(renderedPrompt)
                .systemPrompt(null)
                .maxNewTokens(maxNewTokens)
                .build();
        try (GenerationModelHandle handle =
                GenerationRuntime.load(descriptor, modelDir, backend, LoadPolicy.PACKAGE_ONLY)) {
            return listener == null ? handle.generate(request) : handle.generate(request, listener);
        }
    }

    /**
     * Render the family-correct prompt for {@code promptTask} + {@code userPrompt} by delegating to the
     * shared {@link PromptStrategies} pipeline (T5 prefix, ChatML/Gemma user-turn, Phi-3 system-turn, …).
     * Package-visible so the workbench-to-runtime idempotency can be asserted without loading a model.
     */
    static String renderPrompt(String modelRepositoryId, PromptTask promptTask, String userPrompt) {
        PromptTask task = promptTask == null ? PromptTask.NONE : promptTask;
        return PromptStrategies.forModel(modelRepositoryId).renderPrompt(PromptInput.of(task, userPrompt));
    }

    /** Map the workbench's runtime facade backend onto the neutral catalog backend. */
    public static CatalogBackend toCatalogBackend(Backend backend) {
        switch (backend) {
            case CPU:
                return CatalogBackend.CPU;
            case WARP:
                return CatalogBackend.WARP;
            case DIRECTML:
                return CatalogBackend.DIRECTML;
            case AUTO:
            case HYBRID:
            default:
                // HYBRID has no neutral equivalent; AUTO covers "best available".
                return CatalogBackend.AUTO;
        }
    }
}
