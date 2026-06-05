package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.SafeTensorsReader;
import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.inference.model.TensorEntry;
import com.aresstack.windirectml.inference.model.TensorStorageKind;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * T5-specific output of the model import layer.
 *
 * <p>Keep all foreign source details on this side of the compiler boundary.
 * The runtime must later consume only wdmlpack metadata and runtime tensor
 * handles, never Hugging Face, ONNX, or SafeTensors names directly.</p>
 */
record T5ModelImport(
        String sourceFormat,
        Path modelPath,
        Map<String, OnnxTensor> inlineTensors,
        TensorCatalog tensorCatalog
) {
    static T5ModelImport loadSafeTensorsDirectory(Path modelDir) throws IOException {
        Path root = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
        if (!Files.isDirectory(root)) {
            throw new IOException("T5 SafeTensors model directory not found: " + root);
        }
        List<Path> tensorFiles = discoverSafeTensors(root);
        if (tensorFiles.isEmpty()) {
            throw new IOException("No .safetensors files found in " + root);
        }

        Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
        List<TensorEntry> entries = new ArrayList<>();
        for (Path tensorFile : tensorFiles) {
            SafeTensorsReader.SafeTensorsFile parsed = SafeTensorsReader.read(tensorFile);
            for (SafeTensorsReader.SafeTensorEntry entry : parsed.tensors().values()) {
                if (tensors.containsKey(entry.name())) {
                    throw new IOException("Duplicate SafeTensors tensor name across shards: " + entry.name());
                }
                tensors.put(entry.name(), new OnnxTensor(entry.name(), entry.shape(), entry.onnxDataType(),
                        new float[0], new byte[0], entry.dataBuffer(), Math.toIntExact(entry.byteLength())));
                entries.add(new TensorEntry(entry.name(), entry.onnxDataType(), entry.shape(),
                        TensorStorageKind.INLINE, entry.byteLength()));
            }
        }
        return new T5ModelImport("safetensors", tensorFiles.size() == 1 ? tensorFiles.get(0) : root,
                tensors, new TensorCatalog(entries));
    }

    static List<Path> discoverSafeTensors(Path modelDir) throws IOException {
        Path root = Objects.requireNonNull(modelDir, "modelDir").toAbsolutePath().normalize();
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
