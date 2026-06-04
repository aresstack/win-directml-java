package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.ModelSource;
import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.inference.model.TensorEntry;
import com.aresstack.windirectml.inference.model.TensorStorageKind;
import com.aresstack.windirectml.windows.OnnxModelReader;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * ONNX-backed Qwen model source. ONNX stays behind this import boundary.
 */
final class QwenOnnxModelSource implements ModelSource<QwenModelImport> {

    private static final Logger log = LoggerFactory.getLogger(QwenOnnxModelSource.class);

    private final Path modelDir;
    private final String modelFileName;
    private final Path modelPath;

    QwenOnnxModelSource(Path modelDir, String modelFileName) {
        this.modelDir = modelDir;
        this.modelFileName = QwenModelDirValidator.normalizeModelFileName(modelFileName);
        this.modelPath = modelDir.resolve(this.modelFileName);
    }

    @Override
    public String format() {
        return "onnx";
    }

    @Override
    public Path location() {
        return modelPath;
    }

    @Override
    public QwenModelImport load() throws IOException {
        if (!Files.exists(modelPath)) {
            throw new IOException("Required file missing: " + modelFileName + " (looked in " + modelDir + ")");
        }

        OnnxGraph graph = OnnxModelReader.parse(modelPath);
        Map<String, Qwen2Weights.ExternalTensorRef> externalRefs = Qwen2Weights.parseExternalRefs(modelPath);
        Map<String, OnnxTensor> inlineTensors = graph.initializers();
        TensorCatalog catalog = buildCatalog(inlineTensors, externalRefs);

        log.info("Qwen model source imported: format={}, file={}, graphNodes={}, {}",
                format(), modelFileName, graph.nodes().size(), catalog.summary());

        return new QwenModelImport(format(), modelPath, graph, externalRefs, inlineTensors, catalog);
    }

    private static TensorCatalog buildCatalog(Map<String, OnnxTensor> inlineTensors,
                                              Map<String, Qwen2Weights.ExternalTensorRef> externalRefs) {
        List<TensorEntry> entries = new ArrayList<>();
        for (Map.Entry<String, OnnxTensor> entry : inlineTensors.entrySet()) {
            String name = entry.getKey();
            Qwen2Weights.ExternalTensorRef external = externalRefs.get(name);
            if (external != null) {
                entries.add(new TensorEntry(name, external.dataType(), external.dims(),
                        TensorStorageKind.EXTERNAL, external.length()));
                continue;
            }
            OnnxTensor tensor = entry.getValue();
            long rawBytes = tensor.rawBytes() != null ? tensor.rawBytes().length : 0L;
            long floatBytes = tensor.data() != null ? (long) tensor.data().length * Float.BYTES : 0L;
            long bytes = rawBytes > 0 ? rawBytes : floatBytes;
            TensorStorageKind kind = bytes > 0 ? TensorStorageKind.INLINE : TensorStorageKind.METADATA_ONLY;
            entries.add(new TensorEntry(name, tensor.dataType(), tensor.dims(), kind, bytes));
        }
        for (Qwen2Weights.ExternalTensorRef ref : externalRefs.values()) {
            if (!inlineTensors.containsKey(ref.name())) {
                entries.add(new TensorEntry(ref.name(), ref.dataType(), ref.dims(),
                        TensorStorageKind.EXTERNAL, ref.length()));
            }
        }
        return new TensorCatalog(entries);
    }
}
