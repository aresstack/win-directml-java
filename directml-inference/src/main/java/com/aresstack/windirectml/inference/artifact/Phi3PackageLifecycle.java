package com.aresstack.windirectml.inference.artifact;

import com.aresstack.windirectml.inference.phi3.Phi3CompileOptions;
import com.aresstack.windirectml.inference.phi3.Phi3RuntimePackage;
import com.aresstack.windirectml.inference.phi3.Phi3WdmlPackCompiler;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Artifact lifecycle for the Phi-3 family (PHI3-WORKBENCH-RUNNABLE-1). Wraps the Phi-3 ONNX-&gt;wdmlpack compiler
 * ({@link Phi3WdmlPackCompiler}) and the package loader ({@link Phi3RuntimePackage}), replacing the previous
 * {@code CompilerMissingLifecycle(PHI3)}: Phi-3 is now downloadable AND convertible to {@code model_phi3.wdmlpack},
 * and a compiled package is detected as a valid runtime package.
 *
 * <p>READY detection uses the <b>light</b> structural {@link Phi3RuntimePackage#open} (header + manifest + tensor
 * directory) -- it does <b>not</b> reconstruct the full in-memory weights (~3 GB). The Workbench runtime status for
 * Phi-3 stays {@code PLANNED} until a heap-light runtime loader lands (see {@code docs/phi3-wdmlpack-compiler-plan.md});
 * this lifecycle only makes the Download/Convert + package-validity path real.</p>
 */
public final class Phi3PackageLifecycle implements ModelPackageLifecycle {

    private static final String PACKAGE_FILE = Phi3CompileOptions.DEFAULT_OUTPUT_NAME;

    @Override
    public ModelFamily family() {
        return ModelFamily.PHI3;
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
        // Raw = the convertible ONNX source. The tokenizer is an inference-time asset validated separately.
        RawAssetInspection.Result raw = RawAssetInspection.inspect(dir,
                List.of("config.json"),
                List.of(List.of("*.onnx", "model.safetensors")));
        Optional<Path> existing = existingPackage(dir);
        PackageState packageState;
        boolean executable = false;
        String reason;
        if (existing.isEmpty()) {
            packageState = PackageState.PACKAGE_MISSING;
            reason = "no runtime package (*.wdmlpack) present";
        } else {
            try {
                // Light structural open only (no full weights() reconstruction).
                Phi3RuntimePackage.open(existing.get());
                packageState = PackageState.PACKAGE_VALID;
                executable = true;
                reason = "runtime-loadable, executable Phi-3 package";
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
        Phi3WdmlPackCompiler.Phi3CompileResult result =
                Phi3WdmlPackCompiler.compile(new Phi3CompileOptions(dir, output, force));
        boolean ok;
        try {
            Phi3RuntimePackage.open(result.output());
            ok = true;
        } catch (Exception e) {
            ok = false;
        }
        return new ModelConversionResult(ok, result.output(),
                "Phi-3 compile: written=" + Files.isRegularFile(result.output())
                        + ", tensors=" + result.tensorCount() + ", payloadBytes=" + result.payloadBytes()
                        + ", runtimeLoadable=" + ok);
    }

    @Override
    public Optional<Path> existingPackage(Path modelDir) {
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
