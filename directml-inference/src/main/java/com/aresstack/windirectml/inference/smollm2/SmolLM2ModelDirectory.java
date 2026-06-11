package com.aresstack.windirectml.inference.smollm2;

import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SourceFingerprint;
import com.aresstack.windirectml.inference.model.SourceTensor;
import com.aresstack.windirectml.inference.model.SourceTensorCatalog;
import com.aresstack.windirectml.inference.model.SourceTensorDataType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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

    /**
     * Aggregates a cheap, metadata-based fingerprint over every SafeTensors file in the
     * directory. SmolLM2 models may ship as several shards, so the aggregate combines the
     * per-file fingerprints (already sorted by name) plus the total size and file list. It
     * lets a built {@code model.wdmlpack} be rejected as stale when its source weights change.
     */
    public SourceAggregate sourceAggregate() throws IOException {
        List<Path> files = safeTensorsFiles();
        if (files.isEmpty()) {
            throw new IOException("no SafeTensors files found: " + root);
        }
        List<String> fileNames = new ArrayList<>(files.size());
        StringBuilder fingerprint = new StringBuilder();
        long sizeBytes = 0L;
        for (Path file : files) {
            SourceFingerprint fp = SourceFingerprint.read(file);
            fileNames.add(fp.fileName());
            sizeBytes += Math.max(fp.sizeBytes(), 0L);
            if (fingerprint.length() > 0) {
                fingerprint.append(';');
            }
            fingerprint.append(fp.value());
        }
        return new SourceAggregate(files.size(), sizeBytes, List.copyOf(fileNames), fingerprint.toString());
    }

    /**
     * Aggregated source identity for the SafeTensors shards backing a SmolLM2 model directory.
     */
    public record SourceAggregate(int fileCount, long sizeBytes, List<String> fileNames, String fingerprint) {
        public SourceAggregate {
            fileNames = List.copyOf(fileNames);
            fingerprint = fingerprint == null ? "" : fingerprint;
        }

        public Map<String, Object> toManifest() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("format", "safetensors");
            out.put("fileCount", fileCount);
            out.put("fileNames", List.copyOf(fileNames));
            out.put("sizeBytes", sizeBytes);
            out.put("fingerprint", fingerprint);
            return out;
        }
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
