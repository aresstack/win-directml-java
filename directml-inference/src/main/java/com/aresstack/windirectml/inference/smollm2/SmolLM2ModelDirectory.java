package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SourceTensor;
import com.aresstack.windirectml.inference.model.SourceTensorCatalog;
import com.aresstack.windirectml.inference.model.SourceTensorDataType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * Reads a Hugging Face SmolLM2 model directory for compile-time analysis.
 */
public final class SmolLM2ModelDirectory {

    private final Path root;
    private final SmolLM2ConfigReader configReader;

    public SmolLM2ModelDirectory(Path root) {
        this(root, new SmolLM2ConfigReader());
    }

    SmolLM2ModelDirectory(Path root, SmolLM2ConfigReader configReader) {
        this.root = Objects.requireNonNull(root, "root").toAbsolutePath().normalize();
        this.configReader = Objects.requireNonNull(configReader, "configReader");
    }

    public Path root() {
        return root;
    }

    public SmolLM2Config readConfig() throws IOException {
        return configReader.read(root.resolve("config.json"));
    }

    public SourceTensorCatalog readTensorCatalog() throws IOException {
        List<Path> files = safeTensorsFiles();
        if (files.isEmpty()) {
            throw new IOException("no SafeTensors files found: " + root);
        }
        List<SourceTensor> tensors = new ArrayList<>();
        for (Path file : files) {
            SafeTensorsReader.SafeTensorsFile safeTensorsFile = SafeTensorsReader.read(file);
            for (SafeTensorsReader.SafeTensorEntry entry : safeTensorsFile.tensors().values()) {
                tensors.add(SourceTensor.inline(
                        entry.name(),
                        SourceTensorDataType.fromSafeTensors(entry.dtype(), entry.onnxDataType()),
                        entry.shape(),
                        entry.byteLength(),
                        entry.dataBuffer()));
            }
        }
        return new SourceTensorCatalog(tensors);
    }

    public List<Path> safeTensorsFiles() throws IOException {
        if (!Files.isDirectory(root)) {
            throw new IOException("SmolLM2 model directory not found: " + root);
        }
        try (Stream<Path> stream = Files.list(root)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(".safetensors"))
                    .sorted(Comparator.comparing(path -> path.getFileName().toString()))
                    .toList();
        }
    }
}
