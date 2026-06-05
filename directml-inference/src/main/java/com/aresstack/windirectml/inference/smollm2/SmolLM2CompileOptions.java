package com.aresstack.windirectml.inference.smollm2;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Parsed SmolLM2 compile-tool options.
 */
public record SmolLM2CompileOptions(Path modelDir, Path output, boolean dryRun, boolean force) {
    public SmolLM2CompileOptions {
        modelDir = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        output = output == null ? null : output.toAbsolutePath().normalize();
    }
}
