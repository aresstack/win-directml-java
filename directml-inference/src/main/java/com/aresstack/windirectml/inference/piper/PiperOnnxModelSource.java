package com.aresstack.windirectml.inference.piper;

import com.aresstack.windirectml.inference.model.ModelSource;
import com.aresstack.windirectml.windows.OnnxModelReader;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;

/**
 * Import a Piper voice using the existing pure-Java ONNX protobuf reader.
 * Keep ONNX Runtime and Python outside the import and execution classpaths.
 */
public final class PiperOnnxModelSource implements ModelSource<PiperModelImport> {

    private static final Set<String> INPUTS = Set.of("input", "input_lengths", "scales");
    private static final Set<String> OUTPUTS = Set.of("output");

    private final Path modelFile;
    private final Path configFile;

    public PiperOnnxModelSource(Path modelFile) {
        this(modelFile, modelFile.resolveSibling(modelFile.getFileName().toString() + ".json"));
    }

    public PiperOnnxModelSource(Path modelFile, Path configFile) {
        this.modelFile = Objects.requireNonNull(modelFile, "modelFile").toAbsolutePath().normalize();
        this.configFile = Objects.requireNonNull(configFile, "configFile").toAbsolutePath().normalize();
    }

    @Override
    public String format() {
        return "onnx";
    }

    @Override
    public Path location() {
        return modelFile;
    }

    @Override
    public PiperModelImport load() throws IOException {
        if (!Files.isRegularFile(modelFile)) {
            throw new IOException("Missing Piper ONNX voice: " + modelFile);
        }
        PiperVoiceConfig voice = PiperVoiceConfig.load(configFile);
        OnnxGraph graph = OnnxModelReader.parse(modelFile);
        if (!graph.inputNames().containsAll(INPUTS) || !graph.outputNames().containsAll(OUTPUTS)) {
            throw new IOException("Piper VITS graph must accept input, input_lengths, scales and emit output");
        }
        Map<String, Long> operators = new TreeMap<>();
        graph.nodes().forEach(node -> operators.merge(node.opType(), 1L, Long::sum));
        if (!operators.containsKey("Conv") || !operators.containsKey("ConvTranspose")) {
            throw new IOException("Piper VITS graph is missing expected convolution operations");
        }
        return new PiperModelImport(modelFile, voice, graph, operators);
    }
}
