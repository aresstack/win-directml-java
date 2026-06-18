package com.aresstack.windirectml.inference.phi3;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * ONNX import boundary for the Phi-3 wdmlpack compiler (PHI3-WDMLPACK-COMPILER-1).
 *
 * <p>Locates and validates a Phi-3 model directory ({@code config.json}, {@code tokenizer.json},
 * {@code model.onnx}, {@code model.onnx.data}) and loads it into in-memory {@link Phi3Weights} records by reusing the
 * existing {@link Phi3Weights#load} ONNX graph-topology extraction. ONNX stays behind this boundary as a pure
 * weight/graph container; <b>no ONNX Runtime, no Python</b>. The downstream runtime remains Java/DirectML.</p>
 */
public final class Phi3OnnxModelSource {

    private final Path modelDir;

    public Phi3OnnxModelSource(Path modelDir) {
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
    }

    public String format() {
        return "onnx";
    }

    public Path location() {
        return modelDir;
    }

    /** The ONNX weight container the compiler fingerprints/relativizes against. */
    public Path modelOnnxPath() {
        return modelDir.resolve("model.onnx");
    }

    /**
     * Validate the required Phi-3 source artifacts and import config + weights via
     * {@link Phi3Config}/{@link Phi3Weights} (the existing ONNX extraction).
     */
    public Imported load() throws IOException {
        Path configPath = modelDir.resolve("config.json");
        if (!Files.isRegularFile(configPath)) {
            throw new IOException("Missing Phi-3 config.json in " + modelDir);
        }
        if (!Files.isRegularFile(modelDir.resolve("model.onnx"))
                || !Files.isRegularFile(modelDir.resolve("model.onnx.data"))) {
            throw new IOException("Missing Phi-3 ONNX weights (model.onnx + model.onnx.data) in " + modelDir);
        }
        Phi3Config config = Phi3Config.load(configPath);
        Phi3Weights weights = Phi3Weights.load(modelDir, config);
        boolean tokenizerPresent = Files.isRegularFile(modelDir.resolve("tokenizer.json"));
        return new Imported(modelDir, config, weights, tokenizerPresent);
    }

    /** Imported Phi-3 source: config + extracted weights + whether the inference tokenizer is present. */
    public record Imported(Path modelDir, Phi3Config config, Phi3Weights weights, boolean tokenizerPresent) {
    }
}
