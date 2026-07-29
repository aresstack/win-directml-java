package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.Phi3InferenceEngine;

/**
 * Catalog-driven adapter for the Phi-3 family (Phi-3-mini-4k-instruct-onnx). Allowed backends per
 * the catalog matrix: CPU, DIRECTML, AUTO. Phi-3 has no WARP mode in the code and the matrix does
 * not fake one — {@link GenerationRuntime} rejects WARP with {@code UNSUPPORTED_BACKEND}.
 */
final class Phi3GenerationAdapter extends InferenceEngineAdapter {

    @Override
    public CatalogModelFamily family() {
        return CatalogModelFamily.PHI3;
    }

    @Override
    protected InferenceEngine createEngine(GenerationModelContext context) {
        return new Phi3InferenceEngine(
                context.modelDirectory(), DEFAULT_MAX_TOKENS, backendString(context));
    }
}
