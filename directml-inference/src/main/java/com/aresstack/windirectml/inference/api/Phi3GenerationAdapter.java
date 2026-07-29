package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.inference.phi3.Phi3Runtime;
import com.aresstack.windirectml.inference.phi3.Phi3RuntimePackage;
import com.aresstack.windirectml.inference.phi3.Phi3Tokenizer;
import com.aresstack.windirectml.inference.phi3.Phi3Weights;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Catalog-driven adapter for the Phi-3 family (Phi-3-mini-4k-instruct-onnx). Loads exclusively from
 * the compiled {@code model_phi3.wdmlpack} via {@link Phi3RuntimePackage} + {@link Phi3Runtime},
 * exactly like the workbench's package-backed Phi-3 path — never raw ONNX, never ONNX Runtime, never
 * Python.
 *
 * <p>Backend: the package-backed Phi-3 runtime ({@code Phi3Runtime}) currently has only a CPU compute
 * path, so the catalog matrix is limited to {@code CPU}. The DirectML kernels in
 * {@code Phi3InferenceEngine} are the raw-ONNX (non-package) path and are not wired to the package
 * weights yet; until they are, DIRECTML/AUTO are not in the matrix and are rejected as
 * {@code UNSUPPORTED_BACKEND} rather than failing at init (see problems.md).
 */
final class Phi3GenerationAdapter implements FamilyGenerationAdapter {

    @Override
    public CatalogModelFamily family() {
        return CatalogModelFamily.PHI3;
    }

    @Override
    public GenerationModelHandle open(GenerationModelContext context) {
        if (context.backend() != CatalogBackend.CPU) {
            // Defensive: the catalog matrix (CPU only) plus GenerationRuntime's check should already
            // have rejected this. A non-implemented combination is UNSUPPORTED_BACKEND, not a failure.
            throw new GenerationException(GenerationErrorCode.UNSUPPORTED_BACKEND,
                    "Phi-3 package-backed runtime supports only CPU today; got " + context.backend());
        }
        Path tokenizerFile = context.modelDirectory().resolve("tokenizer.json");
        if (!Files.isRegularFile(tokenizerFile)) {
            throw new GenerationException(GenerationErrorCode.MODEL_ASSETS_MISSING,
                    "Phi-3 requires tokenizer.json next to the package: " + tokenizerFile);
        }
        Phi3Weights weights = null;
        try {
            Phi3RuntimePackage pkg = Phi3RuntimePackage.open(context.runtimePackageFile());
            weights = pkg.weights();
            Phi3Tokenizer tokenizer = Phi3Tokenizer.load(tokenizerFile);
            // CPU compute (gpuKernels/gpuPipeline null) — matches the workbench package-backed path.
            Phi3Runtime runtime = new Phi3Runtime(pkg.config(), weights, tokenizer);
            return new Phi3PackageGenerationHandle(
                    context.descriptor(), context.backend(), weights, tokenizer, runtime);
        } catch (IOException e) {
            closeQuietly(weights);
            throw new GenerationException(GenerationErrorCode.PACKAGE_NOT_LOADABLE,
                    "could not open Phi-3 package " + context.runtimePackageFile() + ": " + e.getMessage(), e);
        } catch (RuntimeException e) {
            closeQuietly(weights);
            throw e;
        }
    }

    private static void closeQuietly(Phi3Weights weights) {
        if (weights != null) {
            try {
                weights.close();
            } catch (IOException | RuntimeException ignored) {
                // best-effort
            }
        }
    }
}
