package com.aresstack.windirectml.workbench.download;

/**
 * Selectable ONNX weight file variants available in the Hugging Face
 * onnx-community Qwen2.5-Coder-0.5B-Instruct repository.
 */
public enum QwenOnnxModelVariant {

    DEFAULT_DENSE("Default dense", "model.onnx", true),
    FP16("FP16", "model_fp16.onnx", false),
    INT8("INT8", "model_int8.onnx", false),
    UINT8("UINT8", "model_uint8.onnx", false),
    QUANTIZED("Quantized", "model_quantized.onnx", false),
    Q4("Q4", "model_q4.onnx", false),
    Q4F16("Q4F16", "model_q4f16.onnx", false),
    BNB4("BNB4", "model_bnb4.onnx", false);

    private final String label;
    private final String modelFileName;
    private final boolean externalDataRequired;

    QwenOnnxModelVariant(String label, String modelFileName, boolean externalDataRequired) {
        this.label = label;
        this.modelFileName = modelFileName;
        this.externalDataRequired = externalDataRequired;
    }

    public String label() {
        return label;
    }

    public String modelFileName() {
        return modelFileName;
    }

    public boolean externalDataRequired() {
        return externalDataRequired;
    }

    public String externalDataFileName() {
        return externalDataRequired ? "model.onnx_data" : null;
    }

    public boolean matchesModelFileName(String candidate) {
        return modelFileName.equals(candidate);
    }

    public static QwenOnnxModelVariant fromModelFileName(String modelFileName) {
        if (modelFileName == null || modelFileName.trim().isEmpty()) {
            return Q4F16;
        }
        String cleanName = modelFileName.trim();
        for (QwenOnnxModelVariant variant : values()) {
            if (variant.matchesModelFileName(cleanName)) {
                return variant;
            }
        }
        return Q4F16;
    }

    @Override
    public String toString() {
        return label + " — " + modelFileName;
    }
}
