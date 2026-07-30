package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.inference.InferenceEngine;
import com.aresstack.windirectml.inference.qwen.QwenInferenceEngine;

/**
 * Catalog-driven adapter for the Qwen causal/chat family (Qwen2.5-Coder-0.5B-Instruct). Allowed
 * backends per the catalog matrix: WARP, AUTO, CPU.
 */
final class QwenGenerationAdapter extends InferenceEngineAdapter {

    @Override
    public CatalogModelFamily family() {
        return CatalogModelFamily.QWEN;
    }

    @Override
    protected InferenceEngine createEngine(GenerationModelContext context) {
        // QwenInferenceEngine derives its runtime-package name from the ONNX file name
        // (model_q4f16.onnx -> model_q4f16.wdmlpack). The workbench passes the explicit variant file;
        // mirror that by deriving the ONNX name from the catalog package name so the engine's
        // package resolution matches the catalog's runtimePackageFileName (model_q4f16.wdmlpack).
        // The validator accepts either the ONNX or the runtime-loadable wdmlpack, so package-only
        // loading (wdmlpack present, no ONNX) is honoured.
        String packageName = context.descriptor().runtimePackageFileName();
        String onnxFileName = packageName.endsWith(".wdmlpack")
                ? packageName.substring(0, packageName.length() - ".wdmlpack".length()) + ".onnx"
                : packageName;
        return new QwenInferenceEngine(
                context.modelDirectory(), DEFAULT_MAX_TOKENS, backendString(context), onnxFileName);
    }
}
