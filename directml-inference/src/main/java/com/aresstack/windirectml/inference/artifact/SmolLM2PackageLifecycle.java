package com.aresstack.windirectml.inference.artifact;

import com.aresstack.windirectml.inference.smollm2.SmolLM2CompileOptions;
import com.aresstack.windirectml.inference.smollm2.SmolLM2CompileReport;
import com.aresstack.windirectml.inference.smollm2.SmolLM2ModelDirectory;
import com.aresstack.windirectml.inference.smollm2.SmolLM2RuntimePackage;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WdmlPackCompiler;
import com.aresstack.windirectml.inference.smollm2.SmolLM2WdmlPackManifest;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Artifact lifecycle for SmolLM2. Wraps {@link SmolLM2WdmlPackCompiler} and
 * {@link SmolLM2RuntimePackage}, and carries the staleness logic (compiler/schema version +
 * source fingerprint) that previously lived inline in {@code SmolLM2WorkbenchRuntimeRunner}.
 */
public final class SmolLM2PackageLifecycle implements ModelPackageLifecycle {

    private static final String PACKAGE_FILE = "model.wdmlpack";

    private final SmolLM2WdmlPackCompiler compiler;

    public SmolLM2PackageLifecycle() {
        this(new SmolLM2WdmlPackCompiler());
    }

    public SmolLM2PackageLifecycle(SmolLM2WdmlPackCompiler compiler) {
        this.compiler = compiler;
    }

    @Override
    public ModelFamily family() {
        return ModelFamily.SMOLLM2;
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
                List.of("config.json", "tokenizer.json"),
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
                SmolLM2RuntimePackage runtimePackage = SmolLM2RuntimePackage.open(pkg);
                executable = runtimePackage.executable();
                if (!executable) {
                    packageState = PackageState.PACKAGE_NOT_EXECUTABLE;
                    reason = "package not runtime-loadable: " + runtimePackage.runtimeLoadableReason();
                } else if (runtimePackage.compilerVersion() != SmolLM2WdmlPackManifest.COMPILER_VERSION) {
                    packageState = PackageState.PACKAGE_STALE;
                    reason = "stale compilerVersion " + runtimePackage.compilerVersion()
                            + " (expected " + SmolLM2WdmlPackManifest.COMPILER_VERSION + ")";
                } else if (runtimePackage.schemaVersion() != SmolLM2WdmlPackManifest.SCHEMA_VERSION) {
                    packageState = PackageState.PACKAGE_STALE;
                    reason = "stale schemaVersion " + runtimePackage.schemaVersion()
                            + " (expected " + SmolLM2WdmlPackManifest.SCHEMA_VERSION + ")";
                } else {
                    Optional<String> mismatch = detectSourceMismatch(dir, runtimePackage, raw.state());
                    if (mismatch.isPresent()) {
                        packageState = PackageState.PACKAGE_STALE;
                        reason = mismatch.get();
                    } else {
                        packageState = PackageState.PACKAGE_VALID;
                        reason = "runtime-loadable, executable package";
                    }
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
        SmolLM2CompileReport report = compiler.compile(new SmolLM2CompileOptions(dir, output, false, force));
        boolean ok = report.payloadIncluded() && report.runtimeLoadable();
        return new ModelConversionResult(ok, report.output(),
                "SmolLM2 compile: payloadIncluded=" + report.payloadIncluded()
                        + ", runtimeLoadable=" + report.runtimeLoadable()
                        + " (" + report.runtimeLoadableReason() + ")");
    }

    private static Optional<String> detectSourceMismatch(Path modelDir,
                                                         SmolLM2RuntimePackage runtimePackage,
                                                         RawAssetState rawState) throws IOException {
        if (rawState != RawAssetState.RAW_VALID) {
            // Without a current raw source there is nothing to compare against; the package stands on its own.
            return Optional.empty();
        }
        Optional<String> stored = runtimePackage.sourceFingerprint();
        if (stored.isEmpty()) {
            return Optional.of("package predates source fingerprinting");
        }
        String current = new SmolLM2ModelDirectory(modelDir).sourceAggregate().fingerprint();
        if (!stored.get().equals(current)) {
            return Optional.of("source SafeTensors changed since the package was built");
        }
        return Optional.empty();
    }
}
