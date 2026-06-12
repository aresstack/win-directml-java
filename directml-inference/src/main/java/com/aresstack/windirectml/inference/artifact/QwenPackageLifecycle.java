package com.aresstack.windirectml.inference.artifact;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.qwen.QwenWdmlPackCompileTool;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Artifact lifecycle for Qwen2.5-Coder. Conversion compiles a Hugging Face SafeTensors directory
 * into {@code model.wdmlpack} via {@link QwenWdmlPackCompileTool}; package state is read from the
 * generic {@link RuntimeModelPackage} manifest. Qwen has no separate "executable" flag - a
 * payload-backed, runtime-loadable package is the runnable state.
 */
public final class QwenPackageLifecycle implements ModelPackageLifecycle {

    private static final String PACKAGE_FILE = "model.wdmlpack";

    @Override
    public ModelFamily family() {
        return ModelFamily.QWEN;
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
    public ModelArtifactStatus inspect(Path modelDir) {
        Path dir = modelDir.toAbsolutePath().normalize();
        RawAssetInspection.Result raw = RawAssetInspection.inspect(dir,
                List.of("config.json"),
                List.of(List.of("*.safetensors")));
        Path pkg = defaultPackagePath(dir);
        PackageState packageState;
        boolean executable = false;
        String reason;
        if (!Files.isRegularFile(pkg)) {
            packageState = PackageState.PACKAGE_MISSING;
            reason = "no runtime package at " + pkg.getFileName();
        } else {
            try {
                RuntimeModelPackage modelPackage = RuntimeModelPackage.open(pkg);
                boolean loadable = modelPackage.payloadIncluded() && modelPackage.runtimeLoadable();
                if (loadable) {
                    packageState = PackageState.PACKAGE_VALID;
                    executable = true;
                    reason = "payload-backed, runtime-loadable package (" + modelPackage.runtimeLoadMode() + ")";
                } else if (!modelPackage.payloadIncluded()) {
                    packageState = PackageState.PACKAGE_NOT_LOADABLE;
                    reason = "manifest-only package: no tensor payload included (mode "
                            + modelPackage.runtimeLoadMode() + ")";
                } else {
                    packageState = PackageState.PACKAGE_NOT_LOADABLE;
                    reason = "package not runtime-loadable (mode " + modelPackage.runtimeLoadMode() + ")";
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
        QwenWdmlPackCompileTool.CompileResult result = QwenWdmlPackCompileTool.compileSafeTensorsDirectory(
                new QwenWdmlPackCompileTool.CompileOptions(dir, output, true, false, false, force));
        return new ModelConversionResult(result.runtimeLoadable(), result.output(),
                "Qwen compile: runtimeLoadable=" + result.runtimeLoadable()
                        + ", mode=" + result.runtimeLoadMode()
                        + ", tensors=" + result.tensorCount());
    }
}
