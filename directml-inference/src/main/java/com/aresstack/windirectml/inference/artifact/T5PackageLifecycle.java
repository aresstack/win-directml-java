package com.aresstack.windirectml.inference.artifact;

import com.aresstack.windirectml.inference.t5.T5CompileOptions;
import com.aresstack.windirectml.inference.t5.T5InferenceEngine;
import com.aresstack.windirectml.inference.t5.T5RuntimePackage;
import com.aresstack.windirectml.inference.t5.T5WdmlPackCompiler;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Artifact lifecycle for the T5/CodeT5 seq2seq family. Wraps {@link T5WdmlPackCompiler} and
 * {@link T5RuntimePackage} (honest {@code weightsLoadable/runtimeLoadable/executable} status).
 */
public final class T5PackageLifecycle implements ModelPackageLifecycle {

    private static final String PACKAGE_FILE = T5InferenceEngine.DEFAULT_PACKAGE_NAME;

    @Override
    public ModelFamily family() {
        return ModelFamily.T5;
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
                List.of(List.of("*.safetensors", "pytorch_model.bin"),
                        List.of("spiece.model", "tokenizer.json")));
        Optional<Path> existing = findExistingPackage(dir);
        PackageState packageState;
        boolean executable = false;
        String reason;
        if (existing.isEmpty()) {
            packageState = PackageState.PACKAGE_MISSING;
            reason = "no runtime package (*.wdmlpack) present";
        } else {
            try {
                T5RuntimePackage runtimePackage = T5RuntimePackage.open(existing.get());
                executable = runtimePackage.executable();
                if (!runtimePackage.runtimeLoadable()) {
                    packageState = PackageState.PACKAGE_NOT_LOADABLE;
                    reason = "package not runtime-loadable: " + runtimePackage.loadability().runtimeLoadableReason();
                } else if (!executable) {
                    packageState = PackageState.PACKAGE_NOT_EXECUTABLE;
                    reason = "runtime-loadable but not executable: " + runtimePackage.loadability().runtimeLoadableReason();
                } else {
                    packageState = PackageState.PACKAGE_VALID;
                    reason = "runtime-loadable, executable package";
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
        T5WdmlPackCompiler.T5CompileResult result =
                T5WdmlPackCompiler.compile(new T5CompileOptions(dir, output, false, force));
        boolean ok = result.written()
                && result.runtimePackage() != null
                && result.runtimePackage().runtimeLoadable();
        boolean executable = result.runtimePackage() != null && result.runtimePackage().executable();
        return new ModelConversionResult(ok, result.output(),
                "T5 compile: written=" + result.written()
                        + ", runtimeLoadable=" + ok + ", executable=" + executable);
    }

    private static Optional<Path> findExistingPackage(Path modelDir) {
        Path preferred = modelDir.resolve(PACKAGE_FILE);
        if (Files.isRegularFile(preferred)) {
            return Optional.of(preferred.toAbsolutePath().normalize());
        }
        if (!Files.isDirectory(modelDir)) {
            return Optional.empty();
        }
        try (var stream = Files.list(modelDir)) {
            return stream.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(".wdmlpack"))
                    .map(p -> p.toAbsolutePath().normalize())
                    .sorted()
                    .findFirst();
        } catch (IOException e) {
            return Optional.empty();
        }
    }
}
