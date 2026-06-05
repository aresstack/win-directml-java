package com.aresstack.windirectml.workbench.download;

/**
 * Selectable Qwen source format for the Workbench download panel.
 */
public enum QwenDownloadSource {
    ONNX("ONNX", "Download ONNX files from onnx-community"),
    SAFETENSORS("SafeTensors", "Download dense SafeTensors from the canonical Qwen repository");

    private final String label;
    private final String description;

    QwenDownloadSource(String label, String description) {
        this.label = label;
        this.description = description;
    }

    public String label() {
        return label;
    }

    public String description() {
        return description;
    }

    @Override
    public String toString() {
        return label;
    }
}
