package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.windows.OnnxModelReader;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Qwen-specific layout compiler for Hugging Face SafeTensors imports.
 *
 * <p>The SafeTensors container has no graph, so this class is the place where
 * foreign Hugging Face tensor names are mapped to the fixed Qwen runtime roles.
 * The current v26 step is intentionally conservative: it does not quantize or
 * repack dense weights yet. It proves whether a SafeTensors directory is a
 * complete Qwen2 dense layout and records that contract in the wdmlpack manifest.</p>
 */
final class QwenSafeTensorsLayoutCompiler {

    static final String LAYOUT_SCHEMA = "qwen2-hf-dense-layout-v26";
    static final int COMPILER_VERSION = 26;

    private QwenSafeTensorsLayoutCompiler() {
    }

    static LayoutAnalysis analyze(QwenModelImport imported, Qwen2Config config) {
        Objects.requireNonNull(imported, "imported");
        Objects.requireNonNull(config, "config");
        if (!"safetensors".equals(imported.sourceFormat())) {
            return LayoutAnalysis.notSafeTensors(imported.sourceFormat());
        }

        Analyzer analyzer = new Analyzer(imported.inlineTensors(), config);
        analyzer.requiredMatrix("embedding", "model.embed_tokens.weight", config.vocabSize(), config.hiddenSize());
        analyzer.requiredVector("final_norm", "model.norm.weight", config.hiddenSize());

        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            String prefix = "model.layers." + layer;
            analyzer.requiredVector(role(layer, "input_layernorm"),
                    prefix + ".input_layernorm.weight", config.hiddenSize());
            analyzer.requiredVector(role(layer, "post_attention_layernorm"),
                    prefix + ".post_attention_layernorm.weight", config.hiddenSize());
            analyzer.requiredMatrix(role(layer, "self_attn.q_proj"),
                    prefix + ".self_attn.q_proj.weight", config.qSize(), config.hiddenSize());
            analyzer.requiredMatrix(role(layer, "self_attn.k_proj"),
                    prefix + ".self_attn.k_proj.weight", config.kvSize(), config.hiddenSize());
            analyzer.requiredMatrix(role(layer, "self_attn.v_proj"),
                    prefix + ".self_attn.v_proj.weight", config.kvSize(), config.hiddenSize());
            analyzer.requiredMatrix(role(layer, "self_attn.o_proj"),
                    prefix + ".self_attn.o_proj.weight", config.hiddenSize(), config.qSize());
            analyzer.requiredMatrix(role(layer, "mlp.gate_proj"),
                    prefix + ".mlp.gate_proj.weight", config.intermediateSize(), config.hiddenSize());
            analyzer.requiredMatrix(role(layer, "mlp.up_proj"),
                    prefix + ".mlp.up_proj.weight", config.intermediateSize(), config.hiddenSize());
            analyzer.requiredMatrix(role(layer, "mlp.down_proj"),
                    prefix + ".mlp.down_proj.weight", config.hiddenSize(), config.intermediateSize());

            analyzer.optionalVector(role(layer, "self_attn.q_proj.bias"),
                    prefix + ".self_attn.q_proj.bias", config.qSize());
            analyzer.optionalVector(role(layer, "self_attn.k_proj.bias"),
                    prefix + ".self_attn.k_proj.bias", config.kvSize());
            analyzer.optionalVector(role(layer, "self_attn.v_proj.bias"),
                    prefix + ".self_attn.v_proj.bias", config.kvSize());
        }

        if (analyzer.exists("lm_head.weight")) {
            analyzer.optionalMatrix("lm_head", "lm_head.weight", config.vocabSize(), config.hiddenSize());
        } else if (config.tieWordEmbeddings()) {
            analyzer.roles.add(LayoutRole.tied("lm_head", "model.embed_tokens.weight"));
        } else {
            analyzer.missingRequired.add("lm_head.weight");
        }

        return analyzer.finish();
    }

    private static String role(int layer, String role) {
        return "layer." + layer + "." + role;
    }

    record LayoutAnalysis(
            String schema,
            int compilerVersion,
            boolean safeTensorsSource,
            boolean complete,
            boolean runtimeLoadable,
            String runtimeLoadMode,
            int roleCount,
            int tensorCount,
            long payloadBytes,
            List<LayoutRole> roles,
            List<String> missingRequired,
            List<String> shapeErrors,
            List<String> unsupportedRuntimeDtypes
    ) {
        static LayoutAnalysis notSafeTensors(String sourceFormat) {
            return new LayoutAnalysis(LAYOUT_SCHEMA, COMPILER_VERSION, false,
                    false, false, "not-safetensors", 0, 0, 0L,
                    List.of(), List.of("source is not SafeTensors: " + sourceFormat), List.of(), List.of());
        }

        Map<String, Object> toManifest() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("schema", schema);
            out.put("compilerVersion", compilerVersion);
            out.put("sourceLayout", "huggingface-qwen2-dense");
            out.put("safeTensorsSource", safeTensorsSource);
            out.put("complete", complete);
            out.put("runtimeLoadable", runtimeLoadable);
            out.put("runtimeLoadMode", runtimeLoadMode);
            out.put("roleCount", roleCount);
            out.put("tensorCount", tensorCount);
            out.put("payloadBytes", payloadBytes);
            out.put("missingRequired", missingRequired);
            out.put("shapeErrors", shapeErrors);
            out.put("unsupportedRuntimeDtypes", unsupportedRuntimeDtypes);
            out.put("roles", roles.stream().map(LayoutRole::toManifest).toList());
            return out;
        }
    }

    record LayoutRole(String role,
                      String sourceName,
                      String runtimeName,
                      int dataType,
                      String dataTypeName,
                      long[] dims,
                      boolean required,
                      boolean tied) {
        LayoutRole {
            dims = dims == null ? new long[0] : dims.clone();
        }

        static LayoutRole tied(String role, String sourceName) {
            return new LayoutRole(role, sourceName, sourceName, 0, "TIED", new long[0], true, true);
        }

        @Override
        public long[] dims() {
            return dims.clone();
        }

        Map<String, Object> toManifest() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("role", role);
            out.put("sourceName", sourceName);
            out.put("runtimeName", runtimeName);
            out.put("dataType", dataType);
            out.put("dataTypeName", dataTypeName);
            List<Long> shape = new ArrayList<>(dims.length);
            for (long dim : dims) shape.add(dim);
            out.put("dims", shape);
            out.put("required", required);
            out.put("tied", tied);
            return out;
        }
    }

    private static final class Analyzer {
        private final Map<String, OnnxTensor> tensors;
        private final Qwen2Config config;
        private final List<LayoutRole> roles = new ArrayList<>();
        private final List<String> missingRequired = new ArrayList<>();
        private final List<String> shapeErrors = new ArrayList<>();
        private final List<String> unsupportedRuntimeDtypes = new ArrayList<>();
        private long rolePayloadBytes;

        private Analyzer(Map<String, OnnxTensor> tensors, Qwen2Config config) {
            this.tensors = Objects.requireNonNull(tensors, "tensors");
            this.config = Objects.requireNonNull(config, "config");
        }

        private boolean exists(String name) {
            OnnxTensor tensor = tensors.get(name);
            return tensor != null && tensor.rawByteLength() > 0;
        }

        private void requiredMatrix(String role, String name, long rows, long cols) {
            tensor(role, name, true, new long[]{rows, cols});
        }

        private void optionalMatrix(String role, String name, long rows, long cols) {
            tensor(role, name, false, new long[]{rows, cols});
        }

        private void requiredVector(String role, String name, long size) {
            tensor(role, name, true, new long[]{size});
        }

        private void optionalVector(String role, String name, long size) {
            tensor(role, name, false, new long[]{size});
        }

        private void tensor(String role, String name, boolean required, long[] expectedDims) {
            OnnxTensor tensor = tensors.get(name);
            if (tensor == null || tensor.rawByteLength() <= 0) {
                if (required) {
                    missingRequired.add(name);
                }
                return;
            }
            long[] dims = tensor.dims();
            if (!sameDims(dims, expectedDims)) {
                shapeErrors.add(name + " expected " + dimsToString(expectedDims) + " but got " + dimsToString(dims));
                return;
            }
            if (!isCurrentDenseRuntimeDtype(tensor.dataType())) {
                unsupportedRuntimeDtypes.add(name + " uses " + dataTypeName(tensor.dataType())
                        + "; current dense Qwen loader supports FLOAT16/FLOAT only");
            }
            rolePayloadBytes += tensor.rawByteLength();
            roles.add(new LayoutRole(role, name, name, tensor.dataType(), dataTypeName(tensor.dataType()), dims, required, false));
        }

        private LayoutAnalysis finish() {
            boolean complete = missingRequired.isEmpty() && shapeErrors.isEmpty();
            boolean runtimeLoadable = complete && unsupportedRuntimeDtypes.isEmpty();
            String mode = runtimeLoadable
                    ? "wdmlpack-native-dense-payload"
                    : "safetensors-layout-only";
            return new LayoutAnalysis(LAYOUT_SCHEMA, COMPILER_VERSION, true, complete, runtimeLoadable,
                    mode, roles.size(), tensors.size(), rolePayloadBytes, List.copyOf(roles),
                    List.copyOf(missingRequired), List.copyOf(shapeErrors), List.copyOf(unsupportedRuntimeDtypes));
        }

        private static boolean sameDims(long[] actual, long[] expected) {
            if (actual.length != expected.length) return false;
            for (int i = 0; i < actual.length; i++) {
                if (actual[i] != expected[i]) return false;
            }
            return true;
        }
    }

    private static boolean isCurrentDenseRuntimeDtype(int onnxDataType) {
        return onnxDataType == OnnxModelReader.ONNX_FLOAT16
                || onnxDataType == OnnxModelReader.ONNX_FLOAT;
    }

    private static String dataTypeName(int dataType) {
        return switch (dataType) {
            case OnnxModelReader.ONNX_FLOAT -> "FLOAT";
            case OnnxModelReader.ONNX_UINT8 -> "UINT8";
            case OnnxModelReader.ONNX_INT8 -> "INT8";
            case OnnxModelReader.ONNX_INT32 -> "INT32";
            case OnnxModelReader.ONNX_INT64 -> "INT64";
            case OnnxModelReader.ONNX_FLOAT16 -> "FLOAT16";
            case 16 -> "BFLOAT16";
            default -> "ONNX_TYPE_" + dataType;
        };
    }

    private static String dimsToString(long[] dims) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(dims[i]);
        }
        return sb.append(']').toString();
    }
}
