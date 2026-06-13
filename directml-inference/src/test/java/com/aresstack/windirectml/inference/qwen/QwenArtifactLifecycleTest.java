package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.PackageState;
import com.aresstack.windirectml.inference.artifact.QwenPackageLifecycle;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * W3: Qwen honours the artifact lifecycle - the default/strict inference path requires an existing
 * runtime package, never compiles, and never silently falls back to ONNX or auto-creates a manifest.
 */
class QwenArtifactLifecycleTest {

    @TempDir
    Path tempDir;

    private final QwenPackageLifecycle lifecycle = new QwenPackageLifecycle(); // default model.onnx -> model.wdmlpack

    @Test
    void missingPackageIsNotReadyAndValidateThrowsActionableWithoutWriting() throws Exception {
        writeConfig(tempDir, true);
        Files.writeString(tempDir.resolve("model.safetensors"), "x"); // raw present, no package

        ModelArtifactStatus status = lifecycle.inspect(tempDir);
        assertEquals(PackageState.PACKAGE_MISSING, status.packageState());
        assertFalse(status.ready());

        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> lifecycle.validateOrThrowBeforeInference(tempDir));
        assertTrue(error.getMessage().contains("Convert"), error.getMessage());
        assertFalse(hasPackage(tempDir), "inspect/validate must not write a package");
    }

    @Test
    void validPackageIsReadyAfterExplicitConvert() throws Exception {
        writeConfig(tempDir, true);
        writeCompleteDenseQwenSafeTensors(tempDir.resolve("model.safetensors"));
        QwenWdmlPackCompileTool.compileSafeTensorsDirectory(tempDir, tempDir.resolve("model.wdmlpack"));

        ModelArtifactStatus status = lifecycle.inspect(tempDir);
        assertEquals(PackageState.PACKAGE_VALID, status.packageState());
        assertTrue(status.executable());
        assertTrue(status.ready());
        lifecycle.validateOrThrowBeforeInference(tempDir); // must not throw
    }

    @Test
    void strictLoadWithMissingPackageThrowsActionableAndWritesNothing() throws Exception {
        writeConfig(tempDir, true);
        writeCompleteDenseQwenSafeTensors(tempDir.resolve("model.safetensors"));
        Qwen2Config config = Qwen2Config.load(tempDir.resolve("config.json"));

        IOException error = assertThrows(IOException.class,
                () -> Qwen2Weights.load(tempDir, config, QwenModelDirValidator.DEFAULT_MODEL_FILE, /* strict */ true));
        assertTrue(error.getMessage().contains("Use Download tab -> Convert"), error.getMessage());

        assertFalse(hasPackage(tempDir), "strict load must not auto-create a package/manifest");
    }

    private static boolean hasPackage(Path dir) throws IOException {
        try (var stream = Files.list(dir)) {
            return stream.anyMatch(p -> p.getFileName().toString().endsWith(".wdmlpack"));
        }
    }

    // --- minimal Qwen SafeTensors fixtures (mirrors QwenWdmlPackCompileToolTest) ----------------

    private static void writeConfig(Path dir, boolean tieWordEmbeddings) throws Exception {
        String json = "{" +
                "\"hidden_size\":4," +
                "\"num_attention_heads\":1," +
                "\"num_hidden_layers\":1," +
                "\"num_key_value_heads\":1," +
                "\"vocab_size\":4," +
                "\"max_position_embeddings\":16," +
                "\"intermediate_size\":8," +
                "\"rms_norm_eps\":0.000001," +
                "\"rope_theta\":1000000.0," +
                "\"tie_word_embeddings\":" + tieWordEmbeddings +
                "}";
        Files.writeString(dir.resolve("config.json"), json);
    }

    private static void writeCompleteDenseQwenSafeTensors(Path file) throws Exception {
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
        writeSafeTensors(file, tensors);
    }

    private static void writeSafeTensors(Path file, Map<String, long[]> tensors) throws Exception {
        StringBuilder header = new StringBuilder("{");
        int offset = 0;
        boolean first = true;
        for (Map.Entry<String, long[]> entry : tensors.entrySet()) {
            int bytes = elements(entry.getValue()) * Short.BYTES;
            if (!first) header.append(',');
            first = false;
            header.append('"').append(entry.getKey()).append("\":{")
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
