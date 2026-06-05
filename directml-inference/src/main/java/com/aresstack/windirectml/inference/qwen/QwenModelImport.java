package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.SourceTensor;
import com.aresstack.windirectml.inference.model.SourceTensorCatalog;
import com.aresstack.windirectml.inference.model.SourceTensorDataType;
import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * Qwen-specific output of the model import layer.
 *
 * <p>The runtime-facing legacy fields still expose a minimal ONNX-shaped graph
 * because the current Qwen loader consumes that shape. New compiler code should
 * prefer {@link #sourceTensorCatalog()} so ONNX, SafeTensors, and wdmlpack remain
 * import formats behind this boundary.</p>
 */
record QwenModelImport(
        String sourceFormat,
        Path modelPath,
        OnnxGraph graph,
        Map<String, Qwen2Weights.ExternalTensorRef> externalRefs,
        Map<String, OnnxTensor> inlineTensors,
        SourceTensorCatalog sourceTensorCatalog
) {
    QwenModelImport {
        sourceTensorCatalog = Objects.requireNonNull(sourceTensorCatalog, "sourceTensorCatalog");
    }


    QwenModelImport(String sourceFormat,
                    Path modelPath,
                    OnnxGraph graph,
                    Map<String, Qwen2Weights.ExternalTensorRef> externalRefs,
                    Map<String, OnnxTensor> inlineTensors,
                    TensorCatalog tensorCatalog) {
        this(sourceFormat, modelPath, graph, externalRefs, inlineTensors, toSourceTensorCatalog(tensorCatalog));
    }

    private static SourceTensorCatalog toSourceTensorCatalog(TensorCatalog tensorCatalog) {
        Objects.requireNonNull(tensorCatalog, "tensorCatalog");
        return new SourceTensorCatalog(tensorCatalog.entries().values().stream()
                .map(entry -> new SourceTensor(entry.name(),
                        SourceTensorDataType.fromOnnxCode(entry.dataType()),
                        entry.dims(), entry.storageKind(), entry.byteLength(), null))
                .toList());
    }

    TensorCatalog tensorCatalog() {
        return sourceTensorCatalog.toTensorCatalog();
    }
}
