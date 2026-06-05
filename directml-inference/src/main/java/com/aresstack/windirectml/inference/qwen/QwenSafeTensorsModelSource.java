package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.ModelSource;
import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.SourceTensor;
import com.aresstack.windirectml.inference.model.SourceTensorCatalog;
import com.aresstack.windirectml.inference.model.SourceTensorDataType;
import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * SafeTensors-backed Qwen import source.
 *
 * <p>v25 intentionally keeps this as an import/packaging skeleton. It parses
 * SafeTensors tensor directories and exposes them through the same
 * {@link QwenModelImport}/{@link TensorCatalog} seam as ONNX and wdmlpack, but
 * it does not yet claim that raw Hugging Face SafeTensors are directly
 * runtime-loadable. The later compiler step will prepack Qwen tensors into the
 * exact INT4/DirectML layout expected by the current runtime.</p>
 */
final class QwenSafeTensorsModelSource implements ModelSource<QwenModelImport> {

    private static final Logger log = LoggerFactory.getLogger(QwenSafeTensorsModelSource.class);

    private final Path modelDir;
    private final List<Path> tensorFiles;
    private final Qwen2Config config;

    QwenSafeTensorsModelSource(Path modelDir, Qwen2Config config) throws IOException {
        this(modelDir, discoverSafeTensors(modelDir), config);
    }

    QwenSafeTensorsModelSource(Path modelDir, List<Path> tensorFiles, Qwen2Config config) {
        this.modelDir = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        this.tensorFiles = List.copyOf(Objects.requireNonNull(tensorFiles, "tensorFiles"));
        this.config = Objects.requireNonNull(config, "config");
    }

    @Override
    public String format() {
        return "safetensors";
    }

    @Override
    public Path location() {
        return tensorFiles.size() == 1 ? tensorFiles.get(0) : modelDir;
    }

    @Override
    public QwenModelImport load() throws IOException {
        if (tensorFiles.isEmpty()) {
            throw new IOException("No .safetensors files found in " + modelDir);
        }

        Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
        List<SourceTensor> catalogEntries = new ArrayList<>();
        long payloadBytes = 0;
        for (Path tensorFile : tensorFiles) {
            SafeTensorsReader.SafeTensorsFile parsed = SafeTensorsReader.read(tensorFile);
            for (SafeTensorsReader.SafeTensorEntry entry : parsed.tensors().values()) {
                if (tensors.containsKey(entry.name())) {
                    throw new IOException("Duplicate SafeTensors tensor name across shards: " + entry.name());
                }
                tensors.put(entry.name(), new OnnxTensor(
                        entry.name(), entry.shape(), entry.onnxDataType(),
                        new float[0], new byte[0], entry.dataBuffer(), Math.toIntExact(entry.byteLength())));
                catalogEntries.add(SourceTensor.inline(entry.name(),
                        SourceTensorDataType.fromSafeTensors(entry.dtype(), entry.onnxDataType()),
                        entry.shape(), entry.byteLength(), entry.dataBuffer()));
                payloadBytes += entry.byteLength();
            }
        }

        validateQwenSkeleton(tensors);
        SourceTensorCatalog catalog = new SourceTensorCatalog(catalogEntries);
        OnnxGraph graph = new OnnxGraph("safetensors:qwen2", List.of(), tensors, List.of(), List.of());
        log.info("Qwen SafeTensors source imported: files={}, tensors={}, payload={}",
                tensorFiles.size(), tensors.size(), QwenWdmlPackCompiler.formatBytes(payloadBytes));
        return new QwenModelImport(format(), location(), graph, Map.of(), tensors, catalog);
    }

    private void validateQwenSkeleton(Map<String, OnnxTensor> tensors) throws IOException {
        OnnxTensor embedding = tensors.get("model.embed_tokens.weight");
        if (embedding == null) {
            // Hugging Face Qwen2 names normally use this tensor. Keeping the
            // validation strict is useful because it catches accidental use of
            // an unrelated SafeTensors directory before the package compiler runs.
            throw new IOException("SafeTensors Qwen import requires model.embed_tokens.weight");
        }
        long[] dims = embedding.dims();
        if (dims.length != 2 || dims[0] != config.vocabSize() || dims[1] != config.hiddenSize()) {
            throw new IOException("SafeTensors embedding shape mismatch: expected ["
                    + config.vocabSize() + ", " + config.hiddenSize() + "], got " + embeddingShape(dims));
        }
    }

    private static String embeddingShape(long[] dims) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(dims[i]);
        }
        return sb.append(']').toString();
    }

    static List<Path> discoverSafeTensors(Path modelDir) throws IOException {
        Path root = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        if (!Files.isDirectory(root)) {
            throw new IOException("Qwen model directory not found: " + root);
        }
        try (var stream = Files.list(root)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
                    .map(Path::toAbsolutePath)
                    .map(Path::normalize)
                    .sorted()
                    .toList();
        }
    }
}
