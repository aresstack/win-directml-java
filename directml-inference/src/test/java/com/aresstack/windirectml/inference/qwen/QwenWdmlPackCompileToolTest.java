package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.RuntimeModelPackage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class QwenWdmlPackCompileToolTest {

    @TempDir
    Path tempDir;

    @Test
    void compilesCompleteSafeTensorsDirectoryToPayloadPackage() throws Exception {
        writeConfig(tempDir, true);
        writeCompleteDenseQwenSafeTensors(tempDir.resolve("model.safetensors"));
        Path output = tempDir.resolve("qwen.wdmlpack");

        QwenWdmlPackCompileTool.CompileResult result = QwenWdmlPackCompileTool.compileSafeTensorsDirectory(tempDir, output);

        assertEquals(output.toAbsolutePath().normalize(), result.output());
        assertTrue(result.payloadIncluded());
        assertTrue(result.runtimeLoadable());
        assertEquals("wdmlpack-native-dense-payload", result.runtimeLoadMode());
        assertEquals(11, result.tensorCount());
        assertTrue(result.payloadBytes() > 0);

        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(output);
        assertTrue(modelPackage.payloadIncluded());
        assertTrue(modelPackage.runtimeLoadable());
        assertEquals("wdmlpack-native-dense-payload", modelPackage.runtimeLoadMode());
        @SuppressWarnings("unchecked")
        Map<String, Object> layout = (Map<String, Object>) modelPackage.manifest().get("qwenLayout");
        assertNotNull(layout);
        assertEquals("qwen2-hf-dense-layout-v26", layout.get("schema"));
        assertEquals(true, layout.get("complete"));
    }

    @Test
    void rejectsImportOnlySafeTensorsByDefaultButCanWriteAnalysisPackage() throws Exception {
        writeConfig(tempDir, true);
        writeEmbeddingOnlySafeTensors(tempDir.resolve("model.safetensors"));
        Path output = tempDir.resolve("analysis.wdmlpack");

        java.io.IOException rejected = assertThrows(java.io.IOException.class,
                () -> QwenWdmlPackCompileTool.compileSafeTensorsDirectory(tempDir, output));
        assertTrue(rejected.getMessage().contains("not runtime-loadable"));

        QwenWdmlPackCompileTool.CompileResult result = QwenWdmlPackCompileTool.compileSafeTensorsDirectory(
                new QwenWdmlPackCompileTool.CompileOptions(tempDir, output, true, true));

        assertTrue(Files.exists(output));
        assertTrue(result.payloadIncluded());
        assertFalse(result.runtimeLoadable());
        assertEquals("safetensors-layout-only", result.runtimeLoadMode());
    }

    @Test
    void commandLineParserSupportsPositionalModelDirAndOutput() throws Exception {
        writeConfig(tempDir, true);
        writeCompleteDenseQwenSafeTensors(tempDir.resolve("model.safetensors"));
        Path output = tempDir.resolve("cli.wdmlpack");

        int exitCode = QwenWdmlPackCompileTool.run(new String[]{
                tempDir.toString(), "--output", output.toString()
        }, System.out, System.err);

        assertEquals(0, exitCode);
        assertTrue(Files.exists(output));
    }

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

    private static Path writeEmbeddingOnlySafeTensors(Path file) throws Exception {
        Map<String, long[]> tensors = new LinkedHashMap<>();
        tensors.put("model.embed_tokens.weight", new long[]{4, 4});
        return writeSafeTensors(file, tensors);
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
        return writeSafeTensors(file, tensors);
    }

    private static Path writeSafeTensors(Path file, Map<String, long[]> tensors) throws Exception {
        StringBuilder header = new StringBuilder("{");
        int offset = 0;
        boolean first = true;
        for (Map.Entry<String, long[]> entry : tensors.entrySet()) {
            int bytes = elements(entry.getValue()) * Short.BYTES;
            if (!first) header.append(',');
            first = false;
            header.append('"').append(entry.getKey()).append("':{")
                    .append("\"dtype\":\"F16\",")
                    .append("\"shape\":").append(shapeJson(entry.getValue())).append(',')
                    .append("\"data_offsets\":[").append(offset).append(',').append(offset + bytes).append("]}");
            offset += bytes;
        }
        header.append('}');
        String jsonHeader = header.toString().replace("':", "\":");
        byte[] headerBytes = jsonHeader.getBytes(StandardCharsets.UTF_8);
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
