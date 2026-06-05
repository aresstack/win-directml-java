package com.aresstack.windirectml.inference.t5;

import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.inference.model.TensorEntry;
import com.aresstack.windirectml.inference.model.TensorStorageKind;
import com.aresstack.windirectml.windows.OnnxModelReader;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.io.ByteArrayOutputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

final class T5TestFixtures {
    private T5TestFixtures() {
    }

    static T5Config tinyConfig(boolean gated) {
        return new T5Config(List.of("T5ForConditionalGeneration"), "t5", true,
                4, 2, 8, 1, 1, 2, 6,
                4, 16, 1e-6f, 0, 2, 0,
                true, gated ? "gated-gelu" : "relu");
    }

    static T5Config untiedConfig() {
        return new T5Config(List.of("T5ForConditionalGeneration"), "t5", true,
                4, 2, 8, 1, 1, 2, 6,
                4, 16, 1e-6f, 0, 2, 0,
                false, "relu");
    }

    static Map<String, OnnxTensor> completeDenseT5Tensors(T5Config config) {
        Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
        add(tensors, "shared.weight", config.vocabSize(), config.modelSize());
        add(tensors, "encoder.final_layer_norm.weight", config.modelSize());
        add(tensors, "decoder.final_layer_norm.weight", config.modelSize());
        add(tensors, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                config.relativeAttentionBuckets(), config.attentionHeads());
        add(tensors, "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                config.relativeAttentionBuckets(), config.attentionHeads());
        addEncoderBlock(tensors, 0, config);
        addDecoderBlock(tensors, 0, config);
        if (!config.usesTiedWordEmbeddings()) {
            add(tensors, "lm_head.weight", config.vocabSize(), config.modelSize());
        }
        return tensors;
    }

    static T5ModelImport imported(Path path, Map<String, OnnxTensor> tensors) {
        List<TensorEntry> entries = tensors.values().stream()
                .map(tensor -> new TensorEntry(tensor.name(), tensor.dataType(), tensor.dims(),
                        TensorStorageKind.INLINE, tensor.rawByteLength()))
                .toList();
        return new T5ModelImport("safetensors", path, tensors, new TensorCatalog(entries));
    }

    static OnnxTensor tensor(String name, long... dims) {
        return tensor(name, OnnxModelReader.ONNX_FLOAT16, dims);
    }

    static OnnxTensor tensor(String name, int dataType, long... dims) {
        int length = Math.toIntExact(elements(dims) * Short.BYTES);
        ByteBuffer payload = ByteBuffer.allocate(length).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < length; i++) {
            payload.put((byte) (i & 0x7f));
        }
        payload.flip();
        return new OnnxTensor(name, dims, dataType, new float[0], new byte[0], payload.asReadOnlyBuffer(), length);
    }

    static Path writeConfig(Path modelDir, T5Config config) throws IOException {
        Files.createDirectories(modelDir);
        Path configJson = modelDir.resolve("config.json");
        Files.writeString(configJson, "{\n" +
                "  \"architectures\": [\"T5ForConditionalGeneration\"],\n" +
                "  \"model_type\": \"t5\",\n" +
                "  \"is_encoder_decoder\": true,\n" +
                "  \"d_model\": " + config.modelSize() + ",\n" +
                "  \"d_kv\": " + config.keyValueSize() + ",\n" +
                "  \"d_ff\": " + config.feedForwardSize() + ",\n" +
                "  \"num_layers\": " + config.encoderLayers() + ",\n" +
                "  \"num_decoder_layers\": " + config.effectiveDecoderLayers() + ",\n" +
                "  \"num_heads\": " + config.attentionHeads() + ",\n" +
                "  \"vocab_size\": " + config.vocabSize() + ",\n" +
                "  \"relative_attention_num_buckets\": " + config.relativeAttentionBuckets() + ",\n" +
                "  \"relative_attention_max_distance\": " + config.relativeAttentionMaxDistance() + ",\n" +
                "  \"layer_norm_epsilon\": 1e-6,\n" +
                "  \"decoder_start_token_id\": " + config.decoderStartTokenId() + ",\n" +
                "  \"eos_token_id\": " + config.eosTokenId() + ",\n" +
                "  \"pad_token_id\": " + config.padTokenId() + ",\n" +
                "  \"tie_word_embeddings\": " + config.usesTiedWordEmbeddings() + ",\n" +
                "  \"feed_forward_proj\": \"" + config.effectiveFeedForwardProjection() + "\"\n" +
                "}\n", StandardCharsets.UTF_8);
        return configJson;
    }

    static Path writeSafeTensors(Path modelDir, Map<String, OnnxTensor> tensors) throws IOException {
        Files.createDirectories(modelDir);
        Path safeTensors = modelDir.resolve("model.safetensors");
        StringBuilder header = new StringBuilder("{");
        long offset = 0;
        boolean first = true;
        ByteBuffer payload = ByteBuffer.allocate(tensors.values().stream().mapToInt(OnnxTensor::rawByteLength).sum());
        for (OnnxTensor tensor : tensors.values()) {
            if (!first) {
                header.append(',');
            }
            first = false;
            long end = offset + tensor.rawByteLength();
            header.append('\"').append(tensor.name()).append("\":{")
                    .append("\"dtype\":\"").append(dtypeName(tensor.dataType())).append("\",")
                    .append("\"shape\":").append(shapeJson(tensor.dims())).append(',')
                    .append("\"data_offsets\":[").append(offset).append(',').append(end).append("]}");
            ByteBuffer data = tensor.rawDataBuffer();
            data.position(0);
            payload.put(data);
            offset = end;
        }
        header.append('}');
        byte[] headerBytes = header.toString().getBytes(StandardCharsets.UTF_8);
        ByteBuffer file = ByteBuffer.allocate(Long.BYTES + headerBytes.length + payload.position()).order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(headerBytes.length);
        file.put(headerBytes);
        payload.flip();
        file.put(payload);
        file.flip();
        Files.write(safeTensors, file.array());
        return safeTensors;
    }


    static Path writeTorchCheckpoint(Path modelDir, Map<String, OnnxTensor> tensors) throws IOException {
        Files.createDirectories(modelDir);
        Path checkpoint = modelDir.resolve("pytorch_model.bin");
        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(checkpoint))) {
            zip.putNextEntry(new ZipEntry("archive/data.pkl"));
            zip.write(torchStateDictPickle(tensors));
            zip.closeEntry();

            int storageIndex = 0;
            for (OnnxTensor tensor : tensors.values()) {
                zip.putNextEntry(new ZipEntry("archive/data/" + storageIndex));
                ByteBuffer payload = tensor.rawDataBuffer();
                payload.position(0);
                byte[] bytes = new byte[tensor.rawByteLength()];
                payload.get(bytes);
                zip.write(bytes);
                zip.closeEntry();
                storageIndex++;
            }

            zip.putNextEntry(new ZipEntry("archive/version"));
            zip.write("3\n".getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }
        return checkpoint;
    }

    private static byte[] torchStateDictPickle(Map<String, OnnxTensor> tensors) {
        PickleWriter out = new PickleWriter();
        out.proto(2);
        out.emptyDict();
        out.mark();
        int storageIndex = 0;
        for (OnnxTensor tensor : tensors.values()) {
            out.binUnicode(tensor.name());
            out.global("torch._utils", "_rebuild_tensor_v2");
            out.mark();
            out.mark();
            out.binUnicode("storage");
            out.global("torch", torchStorageType(tensor.dataType()));
            out.binUnicode(String.valueOf(storageIndex));
            out.binUnicode("cpu");
            out.binInt(elements(tensor.dims()));
            out.tuple();
            out.binPersId();
            out.binInt(0);
            out.longTuple(tensor.dims());
            out.longTuple(rowMajorStride(tensor.dims()));
            out.newFalse();
            out.emptyDict();
            out.tuple();
            out.reduce();
            storageIndex++;
        }
        out.setItems();
        out.stop();
        return out.toByteArray();
    }

    private static String torchStorageType(int dataType) {
        if (dataType == OnnxModelReader.ONNX_FLOAT16) {
            return "HalfStorage";
        }
        if (dataType == OnnxModelReader.ONNX_FLOAT) {
            return "FloatStorage";
        }
        throw new IllegalArgumentException("Unsupported test torch data type: " + dataType);
    }

    private static long[] rowMajorStride(long[] dims) {
        long[] stride = new long[dims.length];
        long current = 1L;
        for (int index = dims.length - 1; index >= 0; index--) {
            stride[index] = current;
            current *= Math.max(1L, dims[index]);
        }
        return stride;
    }

    private static void addEncoderBlock(Map<String, OnnxTensor> tensors, int layer, T5Config config) {
        add(tensors, "encoder.block." + layer + ".layer.0.layer_norm.weight", config.modelSize());
        add(tensors, "encoder.block." + layer + ".layer.1.layer_norm.weight", config.modelSize());
        addAttention(tensors, "encoder.block." + layer + ".layer.0.SelfAttention", config);
        if (config.usesGatedFeedForward()) {
            add(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wi_0.weight", config.feedForwardSize(), config.modelSize());
            add(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wi_1.weight", config.feedForwardSize(), config.modelSize());
        } else {
            add(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wi.weight", config.feedForwardSize(), config.modelSize());
        }
        add(tensors, "encoder.block." + layer + ".layer.1.DenseReluDense.wo.weight", config.modelSize(), config.feedForwardSize());
    }

    private static void addDecoderBlock(Map<String, OnnxTensor> tensors, int layer, T5Config config) {
        add(tensors, "decoder.block." + layer + ".layer.0.layer_norm.weight", config.modelSize());
        add(tensors, "decoder.block." + layer + ".layer.1.layer_norm.weight", config.modelSize());
        add(tensors, "decoder.block." + layer + ".layer.2.layer_norm.weight", config.modelSize());
        addAttention(tensors, "decoder.block." + layer + ".layer.0.SelfAttention", config);
        addAttention(tensors, "decoder.block." + layer + ".layer.1.EncDecAttention", config);
        if (config.usesGatedFeedForward()) {
            add(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wi_0.weight", config.feedForwardSize(), config.modelSize());
            add(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wi_1.weight", config.feedForwardSize(), config.modelSize());
        } else {
            add(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wi.weight", config.feedForwardSize(), config.modelSize());
        }
        add(tensors, "decoder.block." + layer + ".layer.2.DenseReluDense.wo.weight", config.modelSize(), config.feedForwardSize());
    }

    private static void addAttention(Map<String, OnnxTensor> tensors, String prefix, T5Config config) {
        add(tensors, prefix + ".q.weight", config.attentionInnerSize(), config.modelSize());
        add(tensors, prefix + ".k.weight", config.attentionInnerSize(), config.modelSize());
        add(tensors, prefix + ".v.weight", config.attentionInnerSize(), config.modelSize());
        add(tensors, prefix + ".o.weight", config.modelSize(), config.attentionInnerSize());
    }

    private static void add(Map<String, OnnxTensor> tensors, String name, long... dims) {
        tensors.put(name, tensor(name, dims));
    }

    private static long elements(long[] dims) {
        long total = 1;
        for (long dim : dims) {
            total *= dim;
        }
        return total;
    }

    private static String dtypeName(int dataType) {
        if (dataType == OnnxModelReader.ONNX_FLOAT16) {
            return "F16";
        }
        if (dataType == OnnxModelReader.ONNX_FLOAT) {
            return "F32";
        }
        if (dataType == OnnxModelReader.ONNX_INT8) {
            return "I8";
        }
        if (dataType == OnnxModelReader.ONNX_UINT8) {
            return "U8";
        }
        return "BF16";
    }

    private static String shapeJson(long[] dims) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(dims[i]);
        }
        return sb.append(']').toString();
    }

    private static final class PickleWriter {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        void proto(int version) {
            out.write(0x80);
            out.write(version);
        }

        void mark() {
            out.write('(');
        }

        void emptyDict() {
            out.write('}');
        }

        void tuple() {
            out.write('t');
        }

        void reduce() {
            out.write('R');
        }

        void setItems() {
            out.write('u');
        }

        void binPersId() {
            out.write('Q');
        }

        void newFalse() {
            out.write(0x89);
        }

        void stop() {
            out.write('.');
        }

        void global(String module, String name) {
            out.write('c');
            writeAsciiLine(module);
            writeAsciiLine(name);
        }

        void binInt(long value) {
            if (value >= 0 && value <= 255) {
                out.write('K');
                out.write((int) value);
                return;
            }
            if (value >= Integer.MIN_VALUE && value <= Integer.MAX_VALUE) {
                out.write('J');
                byte[] bytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt((int) value).array();
                out.write(bytes, 0, bytes.length);
                return;
            }
            throw new IllegalArgumentException("Test pickle int is out of range: " + value);
        }

        void binUnicode(String value) {
            byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
            out.write('X');
            out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(bytes.length).array(), 0, 4);
            out.write(bytes, 0, bytes.length);
        }

        void longTuple(long[] values) {
            mark();
            for (long value : values) {
                binInt(value);
            }
            tuple();
        }

        byte[] toByteArray() {
            return out.toByteArray();
        }

        private void writeAsciiLine(String value) {
            byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
            out.write(bytes, 0, bytes.length);
            out.write('\n');
        }
    }
}
