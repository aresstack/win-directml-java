package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.nio.file.Path;
import java.util.Map;

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
}
