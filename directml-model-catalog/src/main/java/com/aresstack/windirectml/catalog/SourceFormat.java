package com.aresstack.windirectml.catalog;

import java.util.Locale;

/**
 * The on-disk source-weight format an install strategy must import before compiling the runtime package.
 *
 * <p>Java-8 compatible.</p>
 */
public enum SourceFormat {

    /** A single {@code model.safetensors} file. */
    SAFETENSORS,
    /** Sharded {@code model-0000x-of-0000y.safetensors} plus a {@code model.safetensors.index.json}. */
    SAFETENSORS_SHARDED,
    /** An ONNX graph ({@code model*.onnx}) with optional external-data sidecar, typically INT4-quantised. */
    ONNX_INT4,
    /** A legacy PyTorch {@code pytorch_model.bin} state dict (restricted import path). */
    TORCH_CHECKPOINT;

    public String token() {
        return name().toLowerCase(Locale.ROOT);
    }
}
