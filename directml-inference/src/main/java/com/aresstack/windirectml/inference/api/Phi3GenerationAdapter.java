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
 * <p>Backends: the certified package-backed compute path is CPU (also used for AUTO, whose contract
 * permits a CPU outcome). Phi-3 has no WARP mode ({@link GenerationRuntime} rejects WARP via the
 * matrix). Explicit {@code DIRECTML} for the package-backed runtime is a named remainder: the only
 * GPU Phi-3 path in the tree today is the raw-ONNX {@code Phi3InferenceEngine}, which is not
 * package-only, so it is intentionally not wired here until the package-backed GPU path is verified
 * on the real model (see docs/LOCAL_ENGINE_CERTIFICATION.md).
 */
final class Phi3GenerationAdapter implements FamilyGenerationAdapter {

    @Override
    public CatalogModelFamily family() {
        return CatalogModelFamily.PHI3;
    }

    @Override
    public GenerationModelHandle open(GenerationModelContext context) {
        if (context.backend() == CatalogBackend.DIRECTML) {
            throw new GenerationException(GenerationErrorCode.INITIALIZATION_FAILED,
                    "Phi-3 package-backed DirectML is not yet wired in the neutral runtime; the "
                            + "certified package-only path is CPU (AUTO also runs CPU). Named remainder "
                            + "pending real-model GPU verification.");
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
