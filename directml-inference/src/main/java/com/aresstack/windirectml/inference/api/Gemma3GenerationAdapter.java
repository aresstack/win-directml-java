package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.inference.gemma.Gemma3NativeWarpRuntime;
import com.aresstack.windirectml.windows.WindowsBindings;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Catalog-driven adapter for Gemma 3 (gemma-3-270m-it). Native Java/DirectML only: WARP (software
 * D3D12 rasterizer) or AUTO (hardware adapter). CPU is deliberately excluded from the catalog matrix
 * (the only Gemma CPU path is an external Python bridge), so {@link GenerationRuntime} rejects CPU
 * with {@code UNSUPPORTED_BACKEND} — this adapter never spawns Python.
 */
final class Gemma3GenerationAdapter implements FamilyGenerationAdapter {

    @Override
    public CatalogModelFamily family() {
        return CatalogModelFamily.GEMMA3;
    }

    @Override
    public GenerationModelHandle open(GenerationModelContext context) {
        WindowsBindings.AdapterMode adapterMode = adapterModeFor(context.backend());
        Path tokenizerJson = context.modelDirectory().resolve("tokenizer.json");
        if (!Files.isRegularFile(tokenizerJson)) {
            throw new GenerationException(GenerationErrorCode.MODEL_ASSETS_MISSING,
                    "Gemma 3 requires tokenizer.json next to the package: " + tokenizerJson);
        }
        // The runtime opens/validates the package and DirectML device lazily on generate(); construction
        // is cheap and holds no native resource.
        Gemma3NativeWarpRuntime runtime =
                new Gemma3NativeWarpRuntime(context.runtimePackageFile(), tokenizerJson, adapterMode);
        return new Gemma3GenerationHandle(context.descriptor(), context.backend(), runtime);
    }

    private static WindowsBindings.AdapterMode adapterModeFor(CatalogBackend backend) {
        switch (backend) {
            case AUTO:
                return WindowsBindings.AdapterMode.HARDWARE;
            case WARP:
                return WindowsBindings.AdapterMode.WARP;
            default:
                // Defensive: the catalog matrix (WARP/AUTO only) plus GenerationRuntime's backend check
                // should already have rejected anything else.
                throw new GenerationException(GenerationErrorCode.UNSUPPORTED_BACKEND,
                        "Gemma 3 supports only WARP or AUTO (native DirectML); got " + backend);
        }
    }
}
