package com.aresstack.windirectml.encoder.pack;

import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.ModelConversionResult;
import com.aresstack.windirectml.inference.artifact.ModelFamily;
import com.aresstack.windirectml.inference.artifact.ModelPackageLifecycle;
import com.aresstack.windirectml.inference.artifact.PackageState;
import com.aresstack.windirectml.inference.artifact.RawAssetInspection;
import com.aresstack.windirectml.inference.model.RuntimeModelPackage;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Artifact lifecycle for BERT-style embedding encoders and cross-encoder rerankers, backed by
 * {@link EncoderWdmlPack}. Makes these families first-class citizens of the homogeneous lifecycle:
 * download -> convert -> run-only-from-package.
 */
public final class EncoderPackageLifecycle implements ModelPackageLifecycle {

    private final ModelFamily family;
    private final String packageFileName;
    private final String packageModelFamily;

    private EncoderPackageLifecycle(ModelFamily family, String packageFileName, String packageModelFamily) {
        this.family = Objects.requireNonNull(family, "family");
        this.packageFileName = Objects.requireNonNull(packageFileName, "packageFileName");
        this.packageModelFamily = Objects.requireNonNull(packageModelFamily, "packageModelFamily");
    }

    /** Lifecycle for MiniLM/E5 embedding encoders ({@code encoder.wdmlpack}). */
    public static EncoderPackageLifecycle embedding() {
        return new EncoderPackageLifecycle(ModelFamily.EMBEDDING,
                EncoderWdmlPack.ENCODER_PACKAGE_FILE, EncoderWdmlPack.FAMILY_ENCODER);
    }

    /** Lifecycle for cross-encoder rerankers ({@code reranker.wdmlpack}). */
    public static EncoderPackageLifecycle reranker() {
        return new EncoderPackageLifecycle(ModelFamily.RERANKER,
                EncoderWdmlPack.RERANKER_PACKAGE_FILE, EncoderWdmlPack.FAMILY_RERANKER);
    }

    @Override
    public ModelFamily family() {
        return family;
    }

    @Override
    public boolean hasCompiler() {
        return true;
    }

    @Override
    public Path defaultPackagePath(Path modelDir) {
        return modelDir.resolve(packageFileName).toAbsolutePath().normalize();
    }

    @Override
    public ModelArtifactStatus inspect(Path modelDir) {
        Path dir = modelDir.toAbsolutePath().normalize();
        RawAssetInspection.Result raw = RawAssetInspection.inspect(dir,
                List.of("config.json", "tokenizer.json", "model.safetensors"), List.of());
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
                if (modelPackage.payloadIncluded() && modelPackage.runtimeLoadable()) {
                    packageState = PackageState.PACKAGE_VALID;
                    executable = true;
                    reason = "payload-backed encoder package (" + modelPackage.runtimeLoadMode() + ")";
                } else {
                    packageState = PackageState.PACKAGE_NOT_LOADABLE;
                    reason = "package not runtime-loadable (mode " + modelPackage.runtimeLoadMode() + ")";
                }
            } catch (Exception e) {
                packageState = PackageState.PACKAGE_CORRUPT;
                reason = "package could not be opened: " + e.getMessage();
            }
        }
        return new ModelArtifactStatus(family, dir, raw.state(), packageState, executable, true, reason);
    }

    @Override
    public ModelConversionResult convert(Path modelDir, boolean force) throws IOException {
        Path dir = modelDir.toAbsolutePath().normalize();
        Path output = defaultPackagePath(dir);
        if (Files.exists(output) && !force) {
            throw new IOException("Output already exists: " + output + " (use force to overwrite)");
        }
        Path written = EncoderWdmlPack.compile(dir, output, packageModelFamily);
        boolean ok;
        try {
            RuntimeModelPackage pkg = RuntimeModelPackage.open(written);
            ok = pkg.payloadIncluded() && pkg.runtimeLoadable();
        } catch (Exception e) {
            ok = false;
        }
        return new ModelConversionResult(ok, written,
                family.displayName() + " compile: runtimeLoadable=" + ok + " -> " + written.getFileName());
    }
}
