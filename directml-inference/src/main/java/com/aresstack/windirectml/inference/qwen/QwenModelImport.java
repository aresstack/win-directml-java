package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.nio.file.Path;
import java.util.Map;

/**
 * Qwen-specific output of the model import layer.
 *
 * <p>The current implementation is backed by ONNX metadata, but callers should
 * depend on this record rather than directly opening ONNX. A future SafeTensors
 * or .wdmlpack importer can create the same runtime-facing structures.</p>
 */
record QwenModelImport(
        String sourceFormat,
        Path modelPath,
        OnnxGraph graph,
        Map<String, Qwen2Weights.ExternalTensorRef> externalRefs,
        Map<String, OnnxTensor> inlineTensors,
        TensorCatalog tensorCatalog
) {
}
