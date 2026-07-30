package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Entry point of the neutral generation runtime.
 *
 * <p>Resolves a catalog {@link LocalRuntimeModelDescriptor} + on-disk model directory + requested
 * {@link CatalogBackend} into an open, package-backed {@link GenerationModelHandle}. Model
 * resolution is catalog-driven ({@code runtimeFamily} / {@code runtimeModelId} /
 * {@code packageLifecycleId} / {@code runtimePackageFileName}); there is no name-fragment guessing
 * and no second model list.
 *
 * <p>Load flow, in order (each step fails closed with a typed {@link GenerationException}):
 * <ol>
 *   <li>argument validation → {@link GenerationErrorCode#INVALID_REQUEST}</li>
 *   <li>backend admissibility against the descriptor's allowed-backend matrix →
 *       {@link GenerationErrorCode#UNSUPPORTED_BACKEND} (e.g. Gemma&nbsp;3 + CPU)</li>
 *   <li>model directory existence → {@link GenerationErrorCode#MODEL_DIRECTORY_NOT_FOUND}</li>
 *   <li>runtime package presence under {@link LoadPolicy#PACKAGE_ONLY} →
 *       {@link GenerationErrorCode#PACKAGE_MISSING}</li>
 *   <li>family adapter lookup → {@link GenerationErrorCode#UNSUPPORTED_FAMILY}</li>
 *   <li>adapter open (package load + init) → adapter-specific error codes</li>
 * </ol>
 */
public final class GenerationRuntime {

    private final GenerationAdapterRegistry registry;

    private GenerationRuntime(GenerationAdapterRegistry registry) {
        this.registry = registry;
    }

    /** A runtime backed by the built-in family adapters. */
    public static GenerationRuntime create() {
        return new GenerationRuntime(GenerationAdapterRegistry.defaults());
    }

    /** Package-visible factory for injecting a custom adapter set in tests. */
    static GenerationRuntime withRegistry(GenerationAdapterRegistry registry) {
        return new GenerationRuntime(registry);
    }

    /**
     * Load a model with the fail-closed production policy ({@link LoadPolicy#PACKAGE_ONLY}).
     * Convenience over {@link #open(LocalRuntimeModelDescriptor, Path, CatalogBackend, LoadPolicy)}.
     */
    public static GenerationModelHandle load(
            LocalRuntimeModelDescriptor descriptor, Path modelDirectory, CatalogBackend backend) {
        return load(descriptor, modelDirectory, backend, LoadPolicy.PACKAGE_ONLY);
    }

    /** Static convenience: build a default runtime and open the model. */
    public static GenerationModelHandle load(
            LocalRuntimeModelDescriptor descriptor,
            Path modelDirectory,
            CatalogBackend backend,
            LoadPolicy loadPolicy) {
        return create().open(descriptor, modelDirectory, backend, loadPolicy);
    }

    /**
     * Open a model handle, running the full validation flow described on this class.
     */
    public GenerationModelHandle open(
            LocalRuntimeModelDescriptor descriptor,
            Path modelDirectory,
            CatalogBackend backend,
            LoadPolicy loadPolicy) {
        if (descriptor == null || modelDirectory == null || backend == null || loadPolicy == null) {
            throw new GenerationException(GenerationErrorCode.INVALID_REQUEST,
                    "descriptor, modelDirectory, backend and loadPolicy are all required");
        }

        if (!descriptor.supportedBackends().contains(backend)) {
            throw new GenerationException(GenerationErrorCode.UNSUPPORTED_BACKEND,
                    "backend " + backend + " is not allowed for family "
                            + descriptor.runtimeFamily() + " / model "
                            + descriptor.runtimeModelId() + "; allowed: "
                            + descriptor.supportedBackends());
        }

        if (!Files.isDirectory(modelDirectory)) {
            throw new GenerationException(GenerationErrorCode.MODEL_DIRECTORY_NOT_FOUND,
                    "model directory does not exist: " + modelDirectory);
        }

        Path runtimePackageFile = modelDirectory.resolve(descriptor.runtimePackageFileName());
        if (loadPolicy == LoadPolicy.PACKAGE_ONLY && !Files.isRegularFile(runtimePackageFile)) {
            throw new GenerationException(GenerationErrorCode.PACKAGE_MISSING,
                    "package-only load requires " + descriptor.runtimePackageFileName()
                            + " in " + modelDirectory + " (not found); compile it first or use "
                            + "LoadPolicy.ALLOW_COMPILE");
        }

        FamilyGenerationAdapter adapter = registry.find(descriptor.runtimeFamily())
                .orElseThrow(() -> new GenerationException(GenerationErrorCode.UNSUPPORTED_FAMILY,
                        "no generation adapter registered for family "
                                + descriptor.runtimeFamily()));

        GenerationModelContext context = new GenerationModelContext(
                descriptor, modelDirectory, runtimePackageFile, backend, loadPolicy);
        try {
            return Objects.requireNonNull(adapter.open(context),
                    "adapter returned a null handle for " + descriptor.runtimeModelId());
        } catch (GenerationException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new GenerationException(GenerationErrorCode.INITIALIZATION_FAILED,
                    "failed to open " + descriptor.runtimeModelId() + " (" + descriptor.runtimeFamily()
                            + ") on " + backend + ": " + e.getMessage(), e);
        }
    }
}
