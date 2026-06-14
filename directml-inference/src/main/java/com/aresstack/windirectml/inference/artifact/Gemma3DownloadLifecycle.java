package com.aresstack.windirectml.inference.artifact;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Download/probe lifecycle for Gemma 3 candidate models.
 *
 * <p>Gemma 3 is intentionally not treated like a broken runtime package. The Workbench can
 * download and validate the gated Hugging Face files today, while the IronMind WARP package
 * compiler/runtime is still planned. This lifecycle therefore reports raw download health and
 * keeps conversion disabled without surfacing it as a compiler failure.</p>
 */
public final class Gemma3DownloadLifecycle implements ModelPackageLifecycle {

    private static final List<String> REQUIRED_FILES = List.of(
            "model.safetensors",
            "config.json",
            "tokenizer.json");

    private static final List<List<String>> ANY_OF_GROUPS = List.of(
            List.of("*.safetensors"));

    @Override
    public ModelFamily family() {
        return ModelFamily.GEMMA3;
    }

    @Override
    public boolean hasCompiler() {
        return false;
    }

    @Override
    public Path defaultPackagePath(Path modelDir) {
        return null;
    }

    @Override
    public ModelArtifactStatus inspect(Path modelDir) {
        RawAssetInspection.Result raw = RawAssetInspection.inspect(modelDir, REQUIRED_FILES, ANY_OF_GROUPS);
        return new ModelArtifactStatus(family(), normalize(modelDir), raw.state(),
                PackageState.PACKAGE_COMPILER_MISSING, false, false, statusReason(raw));
    }

    @Override
    public ModelConversionPlan planConversion(Path modelDir) {
        ModelArtifactStatus status = inspect(modelDir);
        return new ModelConversionPlan(ModelConversionAction.NOT_SUPPORTED, null,
                sourceDescription(modelDir), status.reason());
    }

    @Override
    public ModelConversionResult convert(Path modelDir, boolean force) throws IOException {
        return ModelConversionResult.failed("Gemma 3 is a download/probe candidate. "
                + "The WARP runtime-package compiler is not available yet.");
    }

    @Override
    public String actionableMessage(ModelArtifactStatus status) {
        return status.reason();
    }

    private static String statusReason(RawAssetInspection.Result raw) {
        if (raw.state() == RawAssetState.RAW_VALID) {
            return "Gemma 3 files are present. Runtime integration is planned; download/probe only.";
        }
        return "Gemma 3 download is not complete yet: " + raw.detail();
    }

    private static Path normalize(Path modelDir) {
        return modelDir == null ? Path.of(".").toAbsolutePath().normalize() : modelDir.toAbsolutePath().normalize();
    }
}
