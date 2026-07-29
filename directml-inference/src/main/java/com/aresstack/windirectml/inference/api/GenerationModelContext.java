package com.aresstack.windirectml.inference.api;

import com.aresstack.windirectml.catalog.CatalogBackend;
import com.aresstack.windirectml.catalog.LocalRuntimeModelDescriptor;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Resolved inputs handed to a {@link FamilyGenerationAdapter} when opening a model. The
 * {@link GenerationRuntime} has already validated the backend against the catalog matrix, confirmed
 * the model directory exists, and (under {@link LoadPolicy#PACKAGE_ONLY}) confirmed the runtime
 * package file is present.
 *
 * @param descriptor         the catalog descriptor (family, backends, package name, tokenizer, …)
 * @param modelDirectory     the on-disk directory holding the model assets
 * @param runtimePackageFile the resolved {@code model_*.wdmlpack} path (may not exist under
 *                           {@link LoadPolicy#ALLOW_COMPILE})
 * @param backend            the validated requested backend
 * @param loadPolicy         package-only vs. allow-compile
 */
record GenerationModelContext(
        LocalRuntimeModelDescriptor descriptor,
        Path modelDirectory,
        Path runtimePackageFile,
        CatalogBackend backend,
        LoadPolicy loadPolicy) {

    public GenerationModelContext {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(modelDirectory, "modelDirectory");
        Objects.requireNonNull(runtimePackageFile, "runtimePackageFile");
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(loadPolicy, "loadPolicy");
    }

    public String runtimeModelId() {
        return descriptor.runtimeModelId();
    }
}
