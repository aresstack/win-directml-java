package com.aresstack.windirectml.inference.phi3;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Options for the Phi-3 wdmlpack compiler (PHI3-WDMLPACK-COMPILER-1).
 *
 * @param modelDir directory holding the Phi-3 ONNX source (config.json, model.onnx, model.onnx.data, tokenizer.json)
 * @param output   target {@code .wdmlpack}; {@code null} -> {@code model_phi3.wdmlpack} in {@code modelDir}
 * @param force    overwrite an existing package
 */
public record Phi3CompileOptions(Path modelDir, Path output, boolean force) {

    public static final String DEFAULT_OUTPUT_NAME = "model_phi3.wdmlpack";

    public Phi3CompileOptions {
        Objects.requireNonNull(modelDir, "modelDir");
    }

    public Path resolveOutput() {
        if (output != null) {
            return output.toAbsolutePath().normalize();
        }
        return modelDir.resolve(DEFAULT_OUTPUT_NAME).toAbsolutePath().normalize();
    }
}
