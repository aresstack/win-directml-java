package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.TensorStorageKind;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class QwenSafeTensorsModelSourceTest {

    @TempDir
    Path tempDir;

    @Test
    void importsSafeTensorsIntoQwenTensorCatalog() throws Exception {
        Path safetensors = writeSafeTensors(tempDir.resolve("model.safetensors"));
        Qwen2Config config = new Qwen2Config(2, 1, 1, 1, 4,
                16, 4, 1e-6f, 1_000_000f, true);

        QwenSafeTensorsModelSource source = new QwenSafeTensorsModelSource(tempDir, List.of(safetensors), config);
        QwenModelImport imported = source.load();

        assertEquals("safetensors", imported.sourceFormat());
        assertEquals(safetensors.toAbsolutePath().normalize(), imported.modelPath());
        assertEquals(1, imported.tensorCatalog().size());
        assertEquals(1, imported.sourceTensorCatalog().size());
        assertEquals(TensorStorageKind.INLINE,
                imported.tensorCatalog().get("model.embed_tokens.weight").storageKind());
        assertEquals("FLOAT16", imported.sourceTensorCatalog().get("model.embed_tokens.weight").dataTypeName());
        assertTrue(imported.sourceTensorCatalog().get("model.embed_tokens.weight").hasPayload());
        OnnxTensor tensor = imported.inlineTensors().get("model.embed_tokens.weight");
        assertNotNull(tensor);
        assertEquals(10, tensor.dataType());
        assertEquals(16, tensor.rawByteLength());
        assertEquals(0x11, tensor.rawDataBuffer().get(0));
    }

    @Test
    void safetensorsManifestIsImportOnlyNotRuntimeLoadableYet() throws Exception {
        Path safetensors = writeSafeTensors(tempDir.resolve("model.safetensors"));
        Qwen2Config config = new Qwen2Config(2, 1, 1, 1, 4,
                16, 4, 1e-6f, 1_000_000f, true);
        QwenModelImport imported = new QwenSafeTensorsModelSource(tempDir, List.of(safetensors), config).load();

        Map<String, Object> manifest = QwenWdmlPackCompiler.buildManifest(
                imported, config, tempDir, "model.safetensors");

        assertEquals("safetensors", ((Map<?, ?>) manifest.get("source")).get("format"));
        assertEquals(false, manifest.get("runtimeLoadable"));
        assertEquals("import-only-tensor-catalog", manifest.get("runtimeLoadMode"));
        @SuppressWarnings("unchecked")
        Map<String, Object> qwenLayout = (Map<String, Object>) manifest.get("qwenLayout");
        assertNotNull(qwenLayout);
        assertEquals("qwen2-hf-dense-layout-v28", qwenLayout.get("schema"));
        assertEquals(false, qwenLayout.get("complete"));
        assertEquals(false, qwenLayout.get("runtimeLoadable"));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> tensors = (List<Map<String, Object>>) manifest.get("tensors");
        assertEquals("model.embed_tokens.weight", tensors.get(0).get("name"));
    }

    @Test
    void completeSafeTensorsLayoutBecomesDenseRuntimeLoadablePayloadManifest() throws Exception {
        Path safetensors = writeCompleteDenseQwenSafeTensors(tempDir.resolve("model.safetensors"));
        Qwen2Config config = new Qwen2Config(4, 1, 1, 1, 4,
                16, 8, 1e-6f, 1_000_000f, true);
        QwenModelImport imported = new QwenSafeTensorsModelSource(tempDir, List.of(safetensors), config).load();

        Map<String, Long> offsets = new LinkedHashMap<>();
        long cursor = 0;
        var entries = new ArrayList<>(imported.tensorCatalog().entries().values());
        entries.sort(Comparator.comparing(com.aresstack.windirectml.inference.model.TensorEntry::name));
        for (var entry : entries) {
            offsets.put(entry.name(), cursor);
            cursor += entry.byteLength();
        }
        QwenWdmlPackCompiler.PayloadPlan plan =
                new QwenWdmlPackCompiler.PayloadPlan(offsets, List.of(), cursor);

        Map<String, Object> manifest = QwenWdmlPackCompiler.buildPayloadManifest(
                imported, config, tempDir, "model.safetensors", plan);

        assertEquals(true, manifest.get("runtimeLoadable"));
        assertEquals("wdmlpack-native-dense-payload", manifest.get("runtimeLoadMode"));
        @SuppressWarnings("unchecked")
        Map<String, Object> qwenLayout = (Map<String, Object>) manifest.get("qwenLayout");
        assertNotNull(qwenLayout);
        assertEquals(true, qwenLayout.get("complete"));
        assertEquals(true, qwenLayout.get("runtimeLoadable"));
        assertEquals("wdmlpack-native-dense-payload", qwenLayout.get("runtimeLoadMode"));
        assertEquals(12, qwenLayout.get("roleCount"));
    }

    private static Path writeSafeTensors(Path file) throws Exception {
        String header = "{\"model.embed_tokens.weight\":{" +
                "\"dtype\":\"F16\",\"shape\":[4,2],\"data_offsets\":[0,16]" +
                "}}";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer out = ByteBuffer.allocate(Long.BYTES + headerBytes.length + 16).order(ByteOrder.LITTLE_ENDIAN);
        out.putLong(headerBytes.length);
        out.put(headerBytes);
        byte[] payload = new byte[16];
        for (int i = 0; i < payload.length; i++) payload[i] = (byte) (0x11 + i);
        out.put(payload);
        Files.write(file, out.array());
        return file;
    }


    private static Path writeCompleteDenseQwenSafeTensors(Path file) throws Exception {
        Map<String, long[]> tensors = new LinkedHashMap<>();
        tensors.put("model.embed_tokens.weight", new long[]{4, 4});
        tensors.put("model.norm.weight", new long[]{4});
        tensors.put("model.layers.0.input_layernorm.weight", new long[]{4});
        tensors.put("model.layers.0.post_attention_layernorm.weight", new long[]{4});
        tensors.put("model.layers.0.self_attn.q_proj.weight", new long[]{4, 4});
        tensors.put("model.layers.0.self_attn.k_proj.weight", new long[]{4, 4});
        tensors.put("model.layers.0.self_attn.v_proj.weight", new long[]{4, 4});
        tensors.put("model.layers.0.self_attn.o_proj.weight", new long[]{4, 4});
        tensors.put("model.layers.0.mlp.gate_proj.weight", new long[]{8, 4});
        tensors.put("model.layers.0.mlp.up_proj.weight", new long[]{8, 4});
        tensors.put("model.layers.0.mlp.down_proj.weight", new long[]{4, 8});

        StringBuilder header = new StringBuilder("{");
        int offset = 0;
        boolean first = true;
        for (Map.Entry<String, long[]> entry : tensors.entrySet()) {
            int bytes = elements(entry.getValue()) * Short.BYTES;
            if (!first) header.append(',');
            first = false;
            header.append('\"').append(entry.getKey()).append("\":{")
                    .append("\"dtype\":\"F16\",")
                    .append("\"shape\":").append(shapeJson(entry.getValue())).append(',')
                    .append("\"data_offsets\":[").append(offset).append(',').append(offset + bytes).append("]}");
            offset += bytes;
        }
        header.append('}');
        byte[] headerBytes = header.toString().getBytes(StandardCharsets.UTF_8);
        ByteBuffer out = ByteBuffer.allocate(Long.BYTES + headerBytes.length + offset).order(ByteOrder.LITTLE_ENDIAN);
        out.putLong(headerBytes.length);
        out.put(headerBytes);
        for (int i = 0; i < offset; i++) out.put((byte) (i & 0x7f));
        Files.write(file, out.array());
        return file;
    }

    private static int elements(long[] shape) {
        int total = 1;
        for (long dim : shape) total *= Math.toIntExact(dim);
        return total;
    }

    private static String shapeJson(long[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(shape[i]);
        }
        return sb.append(']').toString();
    }

}
