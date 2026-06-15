package com.aresstack.windirectml.inference.artifact;

import com.aresstack.windirectml.inference.gemma.Gemma3RuntimePackage;
import com.aresstack.windirectml.inference.gemma.Gemma3WdmlPackCompiler;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Artifact lifecycle for Gemma 3 (GEMMA-WORKBENCH-CONVERT-1). Wraps the existing
 * {@link Gemma3WdmlPackCompiler} and {@link Gemma3RuntimePackage} so the Workbench can <b>Convert</b> a
 * downloaded Gemma 3 directory into the runtime package the native WARP runtime expects.
 *
 * <p>Supersedes {@code Gemma3DownloadLifecycle} (download/probe-only, no compiler → Convert disabled) in
 * the product wiring (DownloadPanel + DefaultModelArtifactService). The package name is exactly
 * {@link Gemma3WdmlPackCompiler#DEFAULT_OUTPUT_NAME} ({@code model_gemma3.wdmlpack}), the same file
 * {@code Gemma3NativeWarpRuntime} loads.</p>
 */
public final class Gemma3PackageLifecycle implements ModelPackageLifecycle {

    private static final String PACKAGE_FILE = Gemma3WdmlPackCompiler.DEFAULT_OUTPUT_NAME;

    @Override
    public ModelFamily family() {
        return ModelFamily.GEMMA3;
    }

    @Override
    public boolean hasCompiler() {
        return true;
    }

    @Override
    public Path defaultPackagePath(Path modelDir) {
        return modelDir.resolve(PACKAGE_FILE).toAbsolutePath().normalize();
    }

    @Override
    public Optional<Path> existingPackage(Path modelDir) {
        // Exactly model_gemma3.wdmlpack — the runtime loads this specific name, so we never pick another.
        Path preferred = defaultPackagePath(modelDir);
        return Files.isRegularFile(preferred) ? Optional.of(preferred) : Optional.empty();
    }

    @Override
    public ModelArtifactStatus inspect(Path modelDir) {
        Path dir = modelDir.toAbsolutePath().normalize();
        // Raw = the convertible source the compiler needs (config + safetensors weights). tokenizer.json is
        // an inference-time asset the runtime validates separately, not part of the package source.
        RawAssetInspection.Result raw = RawAssetInspection.inspect(dir,
                List.of("config.json"),
                List.of(List.of("*.safetensors")));
        Optional<Path> existing = existingPackage(dir);
        PackageState packageState;
        boolean executable = false;
        String reason;
        if (existing.isEmpty()) {
            packageState = PackageState.PACKAGE_MISSING;
            reason = "no runtime package (" + PACKAGE_FILE + ") present";
        } else {
            try {
                Gemma3RuntimePackage pkg = Gemma3RuntimePackage.open(existing.get());
                if (pkg.runtimeLoadable()) {
                    packageState = PackageState.PACKAGE_VALID;
                    executable = true;
                    reason = "runtime-loadable Gemma 3 package (" + pkg.runtimeLoadMode() + ")";
                } else {
                    packageState = PackageState.PACKAGE_NOT_LOADABLE;
                    reason = "package not runtime-loadable (" + pkg.runtimeLoadMode() + ")";
                }
            } catch (Exception e) {
                packageState = PackageState.PACKAGE_CORRUPT;
                reason = "package could not be opened: " + e.getMessage();
            }
        }
        return new ModelArtifactStatus(family(), dir, raw.state(), packageState, executable, true, reason);
    }

    @Override
    public ModelConversionResult convert(Path modelDir, boolean force) throws IOException {
        Path dir = modelDir.toAbsolutePath().normalize();
        Path output = defaultPackagePath(dir);
        Gemma3WdmlPackCompiler.CompileResult result = Gemma3WdmlPackCompiler.compile(dir, output, force);
        boolean ok = result.written() && result.runtimeLoadable();
        StringBuilder detail = new StringBuilder("Gemma 3 compile: written=").append(result.written())
                .append(", runtimeLoadable=").append(result.runtimeLoadable())
                .append(", tensors=").append(result.tensorCount());
        if (!result.missing().isEmpty()) {
            detail.append(", missing=").append(result.missing());
        }
        if (!result.shapeErrors().isEmpty()) {
            detail.append(", shapeErrors=").append(result.shapeErrors());
        }
        if (!result.dtypeErrors().isEmpty()) {
            detail.append(", dtypeErrors=").append(result.dtypeErrors());
        }
        return new ModelConversionResult(ok, result.output(), detail.toString());
    }
}
