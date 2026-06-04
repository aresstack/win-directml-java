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
        assertEquals(TensorStorageKind.INLINE,
                imported.tensorCatalog().get("model.embed_tokens.weight").storageKind());
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
        List<Map<String, Object>> tensors = (List<Map<String, Object>>) manifest.get("tensors");
        assertEquals("model.embed_tokens.weight", tensors.get(0).get("name"));
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
}
