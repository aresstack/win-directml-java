package com.aresstack.windirectml.inference.piper;

import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/** Describe an imported Piper VITS source without claiming that every operator is executable. */
public record PiperModelImport(Path modelFile, PiperVoiceConfig voice, OnnxGraph graph,
                               Map<String, Long> operators) {

    public PiperModelImport {
        Objects.requireNonNull(modelFile, "modelFile");
        Objects.requireNonNull(voice, "voice");
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(operators, "operators");
        operators = Collections.unmodifiableMap(new TreeMap<>(operators));
    }

    public int nodeCount() {
        return graph.nodes().size();
    }
}
