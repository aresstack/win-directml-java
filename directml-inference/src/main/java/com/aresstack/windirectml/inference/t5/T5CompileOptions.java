package com.aresstack.windirectml.inference.t5;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Command-line options for the T5 wdmlpack compile/inspect tool.
 */
public record T5CompileOptions(Path modelDir,
                               Path output,
                               boolean dryRun,
                               boolean force) {
    static final String DEFAULT_OUTPUT_NAME = "model_t5.wdmlpack";

    public T5CompileOptions {
        Objects.requireNonNull(modelDir, "modelDir");
    }

    Path resolveOutput() {
        if (output != null) {
            return output.toAbsolutePath().normalize();
        }
        return modelDir.resolve(DEFAULT_OUTPUT_NAME).toAbsolutePath().normalize();
    }
}
