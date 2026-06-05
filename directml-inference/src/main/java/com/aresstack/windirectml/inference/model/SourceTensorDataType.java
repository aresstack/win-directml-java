package com.aresstack.windirectml.inference.model;

import java.util.Locale;

/**
 * Format-neutral tensor element type used by the import layer.
 *
 * <p>The runtime package pipeline still writes ONNX-compatible numeric type
 * codes because existing Qwen runtime code consumes them. This value object is
 * the seam that prevents new importers from depending on ONNX tensor classes
 * while the numeric code remains part of the package contract.</p>
 */
public record SourceTensorDataType(int onnxCode, String name, int bytesPerElement) {

    public static final SourceTensorDataType FLOAT = new SourceTensorDataType(1, "FLOAT", 4);
    public static final SourceTensorDataType UINT8 = new SourceTensorDataType(2, "UINT8", 1);
    public static final SourceTensorDataType INT8 = new SourceTensorDataType(3, "INT8", 1);
    public static final SourceTensorDataType UINT16 = new SourceTensorDataType(4, "UINT16", 2);
    public static final SourceTensorDataType INT16 = new SourceTensorDataType(5, "INT16", 2);
    public static final SourceTensorDataType INT32 = new SourceTensorDataType(6, "INT32", 4);
    public static final SourceTensorDataType INT64 = new SourceTensorDataType(7, "INT64", 8);
    public static final SourceTensorDataType BOOL = new SourceTensorDataType(9, "BOOL", 1);
    public static final SourceTensorDataType FLOAT16 = new SourceTensorDataType(10, "FLOAT16", 2);
    public static final SourceTensorDataType DOUBLE = new SourceTensorDataType(11, "DOUBLE", 8);
    public static final SourceTensorDataType UINT32 = new SourceTensorDataType(12, "UINT32", 4);
    public static final SourceTensorDataType UINT64 = new SourceTensorDataType(13, "UINT64", 8);
    public static final SourceTensorDataType BFLOAT16 = new SourceTensorDataType(16, "BFLOAT16", 2);

    public SourceTensorDataType {
        name = name == null || name.isBlank()
                ? "ONNX_TYPE_" + onnxCode
                : name.toUpperCase(Locale.ROOT);
    }

    public static SourceTensorDataType fromOnnxCode(int onnxCode) {
        return switch (onnxCode) {
            case 1 -> FLOAT;
            case 2 -> UINT8;
            case 3 -> INT8;
            case 4 -> UINT16;
            case 5 -> INT16;
            case 6 -> INT32;
            case 7 -> INT64;
            case 9 -> BOOL;
            case 10 -> FLOAT16;
            case 11 -> DOUBLE;
            case 12 -> UINT32;
            case 13 -> UINT64;
            case 16 -> BFLOAT16;
            default -> new SourceTensorDataType(onnxCode, "ONNX_TYPE_" + onnxCode, 0);
        };
    }

    public static SourceTensorDataType fromSafeTensors(String dtype, int onnxCode) {
        String normalized = dtype == null ? "" : dtype.toUpperCase(Locale.ROOT);
        return switch (normalized) {
            case "F32" -> FLOAT;
            case "F16" -> FLOAT16;
            case "BF16" -> BFLOAT16;
            case "F64" -> DOUBLE;
            case "I8" -> INT8;
            case "U8" -> UINT8;
            case "I16" -> INT16;
            case "U16" -> UINT16;
            case "I32" -> INT32;
            case "U32" -> UINT32;
            case "I64" -> INT64;
            case "U64" -> UINT64;
            case "BOOL" -> BOOL;
            default -> fromOnnxCode(onnxCode);
        };
    }

    public boolean isDenseRuntimeFloat() {
        return this.equals(FLOAT) || this.equals(FLOAT16);
    }
}
