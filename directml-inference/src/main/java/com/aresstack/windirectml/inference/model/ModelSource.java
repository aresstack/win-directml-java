package com.aresstack.windirectml.inference.model;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Import/preprocessing boundary for model formats.
 *
 * <p>The execution runtime should load a model-specific import result from this
 * abstraction instead of depending on ONNX, SafeTensors, or any other concrete
 * source format. ONNX is therefore only an import format; later a SafeTensors or
 * wdmlpack source can implement the same boundary without changing the compute
 * path.</p>
 */
public interface ModelSource<T> {

    /**
     * Human-readable format name such as {@code onnx} or {@code safetensors}.
     */
    String format();

    /**
     * The primary file or directory backing this source.
     */
    Path location();

    /**
     * Load and validate source metadata into a model-specific import result.
     */
    T load() throws IOException;
}
