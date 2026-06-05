package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.SourceTensorCatalog;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Compile-time SmolLM2 analyzer and manifest-only package writer.
 */
public final class SmolLM2WdmlPackCompiler {

    private final SmolLM2LayoutValidator layoutValidator;
    private final SmolLM2WdmlPackManifestWriter manifestWriter;

    public SmolLM2WdmlPackCompiler() {
        this(new SmolLM2LayoutValidator(), new SmolLM2WdmlPackManifestWriter());
    }

    SmolLM2WdmlPackCompiler(SmolLM2LayoutValidator layoutValidator,
                            SmolLM2WdmlPackManifestWriter manifestWriter) {
        this.layoutValidator = Objects.requireNonNull(layoutValidator, "layoutValidator");
        this.manifestWriter = Objects.requireNonNull(manifestWriter, "manifestWriter");
    }

    public SmolLM2CompileReport analyze(SmolLM2CompileOptions options) throws IOException {
        SmolLM2ModelDirectory modelDirectory = new SmolLM2ModelDirectory(options.modelDir());
        SmolLM2Config config = modelDirectory.readConfig();
        validateSupported(config);
        SourceTensorCatalog catalog = modelDirectory.readTensorCatalog();
        SmolLM2LayoutReport layoutReport = layoutValidator.validate(config, catalog);
        SmolLM2RuntimeLoadability loadability = SmolLM2RuntimeLoadability.forPackage(layoutReport, false);
        return new SmolLM2CompileReport(resolveOutput(options), true, false, loadability.runtimeLoadable(),
                loadability.runtimeLoadableReason(), config, layoutReport);
    }

    public SmolLM2CompileReport compile(SmolLM2CompileOptions options) throws IOException {
        SmolLM2ModelDirectory modelDirectory = new SmolLM2ModelDirectory(options.modelDir());
        SmolLM2Config config = modelDirectory.readConfig();
        validateSupported(config);
        SourceTensorCatalog catalog = modelDirectory.readTensorCatalog();
        SmolLM2LayoutReport layoutReport = layoutValidator.validate(config, catalog);
        Path output = resolveOutput(options);
        if (Files.exists(output) && !options.force()) {
            throw new IOException("output already exists: " + output + " (use --force to overwrite)");
        }
        boolean payloadIncluded = false;
        if (!options.dryRun()) {
            manifestWriter.writeWithDensePayload(output, config, layoutReport, catalog);
            payloadIncluded = true;
        }
        SmolLM2RuntimeLoadability loadability = SmolLM2RuntimeLoadability.forPackage(layoutReport, payloadIncluded);
        return new SmolLM2CompileReport(output, options.dryRun(), payloadIncluded, loadability.runtimeLoadable(),
                loadability.runtimeLoadableReason(), config, layoutReport);
    }

    private static void validateSupported(SmolLM2Config config) throws IOException {
        SmolLM2ModelFamily family = new SmolLM2ModelFamily();
        if (!family.supports(config)) {
            throw new IOException("unsupported model_type or architecture for SmolLM2: model_type="
                    + config.modelType() + ", architectures=" + config.architectures());
        }
        try {
            family.architecture(config);
        } catch (IllegalArgumentException e) {
            throw new IOException("unsupported SmolLM2 architecture: " + e.getMessage(), e);
        }
    }

    private static Path resolveOutput(SmolLM2CompileOptions options) {
        if (options.output() != null) {
            return options.output();
        }
        return options.modelDir().resolve("model.wdmlpack").toAbsolutePath().normalize();
    }
}
