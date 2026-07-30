package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.CatalogModelFamily;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Runtime;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimePackage;
import com.aresstack.windirectml.inference.smollm2.SmolLM2Tokenizer;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WarpReadinessReport;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WarpRuntime;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Catalog-driven adapter for the SmolLM2 causal family (135M/360M). Ports the workbench's
 * {@code SmolLM2WorkbenchRuntimeRunner} routing into the neutral module:
 * <ul>
 *   <li>CPU  → reference runtime</li>
 *   <li>AUTO → native DirectML on a hardware adapter when WARP is ready, else reference</li>
 *   <li>WARP → native D3D12 WARP software rasterizer when ready, else reference</li>
 * </ul>
 * Loads exclusively from the SmolLM2 {@code model.wdmlpack}; the tokenizer is read from the package
 * directory. AUTO/WARP degrade to the CPU reference path when no usable device is present — that is
 * the documented AUTO/WARP contract, not a silent switch to a foreign backend.
 */
final class SmolLm2GenerationAdapter implements FamilyGenerationAdapter {

    @Override
    public CatalogModelFamily family() {
        return CatalogModelFamily.SMOLLM2;
    }

    @Override
    public GenerationModelHandle open(GenerationModelContext context) {
        Path modelDir = context.modelDirectory();
        Path tokenizerFile = modelDir.resolve("tokenizer.json");
        if (!Files.isRegularFile(tokenizerFile)) {
            throw new GenerationException(GenerationErrorCode.MODEL_ASSETS_MISSING,
                    "SmolLM2 requires tokenizer.json next to the package: " + tokenizerFile);
        }
        Path tokenizerConfig = modelDir.resolve("tokenizer_config.json");
        try {
            SmolLM2RuntimePackage pkg = SmolLM2RuntimePackage.open(context.runtimePackageFile());
            SmolLM2Tokenizer tokenizer = SmolLM2Tokenizer.load(tokenizerFile,
                    Files.isRegularFile(tokenizerConfig) ? tokenizerConfig : null);
            int maxPos = pkg.config().maxPositionEmbeddings();
            Loaded loaded = loadRuntime(context.backend(), pkg, tokenizer, maxPos);
            // Report the ACTUAL backend the runtime resolved to, not the requested one — AUTO may
            // legitimately resolve to CPU, and that must be visible to the caller.
            return new SmolLm2GenerationHandle(context.descriptor(), loaded.actualBackend, loaded.runtime);
        } catch (IOException e) {
            throw new GenerationException(GenerationErrorCode.PACKAGE_NOT_LOADABLE,
                    "could not open SmolLM2 package " + context.runtimePackageFile() + ": " + e.getMessage(), e);
        }
    }

    private static Loaded loadRuntime(CatalogBackend backend, SmolLM2RuntimePackage pkg,
            SmolLM2Tokenizer tokenizer, int maxPos) {
        if (backend == CatalogBackend.CPU) {
            return new Loaded(SmolLM2Runtime.loadReference(pkg, tokenizer), CatalogBackend.CPU);
        }
        if (backend == CatalogBackend.AUTO) {
            // AUTO: use a device when ready, otherwise fall back to the CPU reference — and report the
            // backend actually chosen. This is the documented AUTO contract, not a silent switch.
            if (warpReady(pkg, maxPos)) {
                return new Loaded(SmolLM2Runtime.loadAuto(pkg, tokenizer, maxPos, "auto"), CatalogBackend.AUTO);
            }
            return new Loaded(SmolLM2Runtime.loadReference(pkg, tokenizer), CatalogBackend.CPU);
        }
        if (backend == CatalogBackend.WARP) {
            // Explicit WARP must run WARP or fail — never silently fall back to CPU. (WARP is currently
            // withheld from the SmolLM2 catalog matrix; this remains as a fail-closed guard.)
            if (!warpReady(pkg, maxPos)) {
                throw new GenerationException(GenerationErrorCode.UNSUPPORTED_BACKEND,
                        "SmolLM2 WARP requested but the D3D12 WARP adapter is not ready; no silent CPU "
                                + "fallback for an explicit backend");
            }
            return new Loaded(SmolLM2Runtime.loadWarp(pkg, tokenizer, maxPos, "warp"), CatalogBackend.WARP);
        }
        throw new GenerationException(GenerationErrorCode.UNSUPPORTED_BACKEND,
                "SmolLM2 does not support backend " + backend);
    }

    private static final class Loaded {
        final SmolLM2Runtime runtime;
        final CatalogBackend actualBackend;

        Loaded(SmolLM2Runtime runtime, CatalogBackend actualBackend) {
            this.runtime = runtime;
            this.actualBackend = actualBackend;
        }
    }

    private static boolean warpReady(SmolLM2RuntimePackage pkg, int maxPos) {
        try (SmolLM2WarpRuntime warpRuntime = SmolLM2WarpRuntime.prepare(pkg, maxPos)) {
            return SmolLM2WarpReadinessReport.fromRuntime(warpRuntime).executable();
        } catch (RuntimeException e) {
            return false;
        }
    }
}
