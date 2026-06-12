package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.TorchCheckpointModelSource;
import com.aresstack.windirectml.windows.OnnxModelReader;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * T5-specific layout compiler skeleton for Hugging Face SafeTensors imports.
 *
 * <p>Map foreign T5/CodeT5 tensor names into stable runtime roles before a
 * wdmlpack is written. Keep this importer/compiler concern out of the WARP
 * runtime so T5 execution can later load only the internal package.</p>
 */
final class T5SafeTensorsLayoutCompiler {

    static final String LAYOUT_SCHEMA = "t5-hf-dense-layout-v1";
    static final int COMPILER_VERSION = 1;

    private T5SafeTensorsLayoutCompiler() {
    }

    static T5LayoutManifest analyze(T5ModelImport imported, T5Config config) {
        Objects.requireNonNull(imported, "imported");
        Objects.requireNonNull(config, "config");
        if (!"safetensors".equals(imported.sourceFormat())
                && !TorchCheckpointModelSource.FORMAT.equals(imported.sourceFormat())) {
            return T5LayoutManifest.notSafeTensors(imported.sourceFormat());
        }

        Analyzer analyzer = new Analyzer(imported.inlineTensors(), config);
        analyzer.tensor(T5TensorNameMapper.sharedEmbedding(config));
        analyzer.tensor(T5TensorNameMapper.encoderFinalLayerNorm(config));
        analyzer.tensor(T5TensorNameMapper.decoderFinalLayerNorm(config));
        analyzer.tensor(T5TensorNameMapper.encoderRelativeAttentionBias(config));
        analyzer.tensor(T5TensorNameMapper.decoderRelativeAttentionBias(config));

        for (int layer = 0; layer < config.encoderLayers(); layer++) {
            analyzer.tensor(T5TensorNameMapper.encoderLayerNorm(layer, 0, config));
            analyzer.tensor(T5TensorNameMapper.encoderLayerNorm(layer, 1, config));
            analyzer.encoderSelfAttention(layer);
            analyzer.encoderFeedForward(layer);
        }

        for (int layer = 0; layer < config.effectiveDecoderLayers(); layer++) {
            analyzer.tensor(T5TensorNameMapper.decoderLayerNorm(layer, 0, config));
            analyzer.tensor(T5TensorNameMapper.decoderLayerNorm(layer, 1, config));
            analyzer.tensor(T5TensorNameMapper.decoderLayerNorm(layer, 2, config));
            analyzer.decoderSelfAttention(layer);
            analyzer.decoderCrossAttention(layer);
            analyzer.decoderFeedForward(layer);
        }

        analyzer.lmHead();
        return analyzer.finish();
    }

    private static final class Analyzer {
        private final Map<String, OnnxTensor> tensors;
        private final T5Config config;
        private final List<T5TensorRole> roles = new ArrayList<>();
        private final List<String> missingRequired = new ArrayList<>();
        private final List<String> shapeErrors = new ArrayList<>();
        private final List<String> unsupportedRuntimeDtypes = new ArrayList<>();
        private long rolePayloadBytes;

        private Analyzer(Map<String, OnnxTensor> tensors, T5Config config) {
            this.tensors = Objects.requireNonNull(tensors, "tensors");
            this.config = Objects.requireNonNull(config, "config");
        }

        private void encoderSelfAttention(int layer) {
            attention("encoder", layer, false);
        }

        private void decoderSelfAttention(int layer) {
            attention("decoder", layer, false);
        }

        private void decoderCrossAttention(int layer) {
            attention("decoder", layer, true);
        }

        private void attention(String stack, int layer, boolean crossAttention) {
            for (String projection : List.of("q", "k", "v", "o")) {
                if (crossAttention) {
                    tensor(T5TensorNameMapper.decoderCrossAttention(layer, projection, config));
                } else if ("encoder".equals(stack)) {
                    tensor(T5TensorNameMapper.encoderSelfAttention(layer, projection, config));
                } else {
                    tensor(T5TensorNameMapper.decoderSelfAttention(layer, projection, config));
                }
            }
        }

        private void encoderFeedForward(int layer) {
            feedForward(true, layer);
        }

        private void decoderFeedForward(int layer) {
            feedForward(false, layer);
        }

        private void feedForward(boolean encoder, int layer) {
            if (config.usesGatedFeedForward()) {
                tensor(encoder
                        ? T5TensorNameMapper.encoderFeedForward(layer, "wi_0", config)
                        : T5TensorNameMapper.decoderFeedForward(layer, "wi_0", config));
                tensor(encoder
                        ? T5TensorNameMapper.encoderFeedForward(layer, "wi_1", config)
                        : T5TensorNameMapper.decoderFeedForward(layer, "wi_1", config));
            } else {
                tensor(encoder
                        ? T5TensorNameMapper.encoderFeedForward(layer, "wi", config)
                        : T5TensorNameMapper.decoderFeedForward(layer, "wi", config));
            }
            tensor(encoder
                    ? T5TensorNameMapper.encoderFeedForward(layer, "wo", config)
                    : T5TensorNameMapper.decoderFeedForward(layer, "wo", config));
        }

        private void lmHead() {
            T5ExpectedTensor lmHead = T5TensorNameMapper.lmHead(config);
            if (exists(lmHead.sourceName())) {
                tensor(lmHead);
            } else if (config.usesTiedWordEmbeddings()) {
                roles.add(T5TensorRole.tied("lm_head", "shared.weight", "lm_head.weight"));
            } else {
                missingRequired.add("lm_head.weight");
            }
        }

        private boolean exists(String name) {
            OnnxTensor tensor = tensors.get(name);
            return tensor != null && tensor.rawByteLength() > 0;
        }

        private void tensor(T5ExpectedTensor expected) {
            OnnxTensor tensor = tensors.get(expected.sourceName());
            if (tensor == null || tensor.rawByteLength() <= 0) {
                if (expected.required()) {
                    missingRequired.add(expected.sourceName());
                }
                return;
            }
            long[] dims = tensor.dims();
            if (!sameDims(dims, expected.expectedDims())) {
                shapeErrors.add(expected.sourceName() + " expected "
                        + dimsToString(expected.expectedDims()) + " but got " + dimsToString(dims));
                return;
            }
            if (!isCurrentDenseRuntimeDtype(tensor.dataType())) {
                unsupportedRuntimeDtypes.add(expected.sourceName() + " uses " + dataTypeName(tensor.dataType())
                        + "; current dense T5 loader supports FLOAT16/FLOAT only");
            }
            rolePayloadBytes += tensor.rawByteLength();
            roles.add(new T5TensorRole(expected.role(), expected.sourceName(), expected.runtimeName(),
                    tensor.dataType(), dataTypeName(tensor.dataType()), dims, expected.required(), false));
        }

        private T5LayoutManifest finish() {
            boolean complete = missingRequired.isEmpty() && shapeErrors.isEmpty();
            // A complete layout with only supported runtime dtypes can produce a runtime-loadable package
            // (weights + structures build); generation itself is not yet certified (executable handled at package level).
            boolean runtimeLoadable = complete && unsupportedRuntimeDtypes.isEmpty();
            String runtimeLoadMode = runtimeLoadable
                    ? T5ManifestPayloadPolicy.MODE_RUNTIME_LOADABLE_NOT_EXECUTABLE
                    : T5ManifestPayloadPolicy.MODE_WEIGHTS_NOT_LOADABLE;
            String reason = runtimeLoadable
                    ? T5ManifestPayloadPolicy.REASON_RUNTIME_LOADABLE_NOT_EXECUTABLE
                    : T5ManifestPayloadPolicy.REASON_WEIGHTS_NOT_LOADABLE;
            return new T5LayoutManifest(LAYOUT_SCHEMA, COMPILER_VERSION, "huggingface-t5-dense", true, complete,
                    runtimeLoadable, runtimeLoadMode, reason,
                    roles.size(), tensors.size(), rolePayloadBytes, List.copyOf(roles),
                    List.copyOf(missingRequired), List.copyOf(shapeErrors), List.copyOf(unsupportedRuntimeDtypes));
        }

        private static boolean sameDims(long[] actual, long[] expected) {
            if (actual.length != expected.length) {
                return false;
            }
            for (int i = 0; i < actual.length; i++) {
                if (actual[i] != expected[i]) {
                    return false;
                }
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
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(dims[i]);
        }
        return sb.append(']').toString();
    }
}
