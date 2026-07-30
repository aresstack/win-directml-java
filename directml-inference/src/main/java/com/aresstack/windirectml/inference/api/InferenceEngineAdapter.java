package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.InferenceException;
import java.util.Locale;

/**
 * Base for the family adapters whose engine implements {@link InferenceEngine} (Qwen, T5, Phi-3).
 * Handles the common open/initialize/close lifecycle and the shared backend-string mapping; each
 * subclass only constructs its family's engine.
 */
abstract class InferenceEngineAdapter implements FamilyGenerationAdapter {

    /** Default construction-time token cap; per-call limits come from the request. */
    protected static final int DEFAULT_MAX_TOKENS = GenerationRequest.DEFAULT_MAX_NEW_TOKENS;

    /**
     * The engine backend string for a context. Mirrors the shipping workbench mapping
     * ({@code backend.name().toLowerCase()}); each engine normalises the value internally
     * ("cpu"/"warp"/"auto"/"directml"). Resolution is catalog-driven — {@link GenerationRuntime}
     * has already rejected any backend outside the family's matrix, so no silent path switch occurs.
     */
    protected static String backendString(GenerationModelContext context) {
        return context.backend().name().toLowerCase(Locale.ROOT);
    }

    /** Construct (but do not initialise) the family engine for this context. */
    protected abstract InferenceEngine createEngine(GenerationModelContext context);

    @Override
    public GenerationModelHandle open(GenerationModelContext context) {
        InferenceEngine engine = createEngine(context);
        try {
            engine.initialize();
        } catch (InferenceException e) {
            safeShutdown(engine);
            throw new GenerationException(GenerationErrorCode.INITIALIZATION_FAILED,
                    "failed to initialize " + context.runtimeModelId() + " on " + context.backend()
                            + ": " + e.getMessage(), e);
        } catch (RuntimeException e) {
            safeShutdown(engine);
            throw e;
        }
        return new InferenceEngineGenerationHandle(engine, context.descriptor(), context.backend());
    }

    private static void safeShutdown(InferenceEngine engine) {
        try {
            engine.shutdown();
        } catch (RuntimeException ignored) {
            // best-effort cleanup on a failed open
        }
    }
}
