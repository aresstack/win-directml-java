package com.aresstack.windirectml.inference.artifact;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import com.aresstack.windirectml.inference.qwen.QwenModelDirValidator;
import com.aresstack.windirectml.inference.qwen.QwenWdmlPackCompileTool;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Artifact lifecycle for Qwen2.5-Coder. Conversion compiles a Hugging Face SafeTensors directory
 * into the runtime package via {@link QwenWdmlPackCompileTool}; package state is read from the
 * generic {@link RuntimeModelPackage} manifest. Qwen has no separate "executable" flag - a
 * payload-backed, runtime-loadable package is the runnable state.
 *
 * <p>The package file name is variant-specific (e.g. {@code model_q4f16.wdmlpack}); it is resolved
 * the same way the loader resolves it ({@link QwenWdmlPackCompileTool#resolvePackagePath}) so the
 * lifecycle checks exactly the package the runtime would load.</p>
 */
public final class QwenPackageLifecycle implements ModelPackageLifecycle {

    private final String modelFileName;

    public QwenPackageLifecycle() {
        this(QwenModelDirValidator.DEFAULT_MODEL_FILE);
    }

    public QwenPackageLifecycle(String modelFileName) {
        this.modelFileName = modelFileName == null || modelFileName.isBlank()
                ? QwenModelDirValidator.DEFAULT_MODEL_FILE : modelFileName;
    }

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
        return QwenWdmlPackCompileTool.resolvePackagePath(modelDir, modelFileName);
    }

    @Override
    public ModelArtifactStatus inspect(Path modelDir) {
        Path dir = modelDir.toAbsolutePath().normalize();
        // Raw = the convertible source. For the q4f16/ONNX runtime path (model_q4f16.onnx) require the
        // ONNX file; for the SafeTensors path require any *.safetensors.
        RawAssetInspection.Result raw = isOnnxSource()
                ? RawAssetInspection.inspect(dir, List.of("config.json", modelFileName), List.of())
                : RawAssetInspection.inspect(dir, List.of("config.json"), List.of(List.of("*.safetensors")));
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
        // Pick the convert path that matches the runtime source: q4f16/ONNX or SafeTensors.
        QwenWdmlPackCompileTool.CompileResult result = isOnnxSource()
                ? QwenWdmlPackCompileTool.compileOnnxDirectory(dir, modelFileName, output, force)
                : QwenWdmlPackCompileTool.compileSafeTensorsDirectory(
                        new QwenWdmlPackCompileTool.CompileOptions(dir, output, true, false, false, force));
        return new ModelConversionResult(result.runtimeLoadable(), result.output(),
                "Qwen compile (" + (isOnnxSource() ? "q4f16/onnx" : "safetensors") + "): runtimeLoadable="
                        + result.runtimeLoadable() + ", mode=" + result.runtimeLoadMode()
                        + ", tensors=" + result.tensorCount());
    }

    private boolean isOnnxSource() {
        return modelFileName.endsWith(".onnx");
    }
}
