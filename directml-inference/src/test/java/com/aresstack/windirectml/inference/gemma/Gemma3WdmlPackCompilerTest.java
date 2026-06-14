package com.aresstack.windirectml.inference.gemma;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** GEMMA-WARP-3: HF Gemma dir -> wdmlpack -> reopen -> load weights -> native forward runs. */
class Gemma3WdmlPackCompilerTest {

    @TempDir
    Path tempDir;

    private static final String CONFIG_JSON = """
            {"model_type":"gemma3_text","architectures":["Gemma3ForCausalLM"],
             "hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,"num_attention_heads":2,
             "num_key_value_heads":1,"head_dim":4,"vocab_size":20,"max_position_embeddings":64,
             "sliding_window":3,"layer_types":["sliding_attention","full_attention"],
             "rope_theta":1000000.0,"rope_local_base_freq":10000.0,"query_pre_attn_scalar":4,
             "rms_norm_eps":1e-06,"hidden_activation":"gelu_pytorch_tanh",
             "bos_token_id":2,"eos_token_id":1,"pad_token_id":0}
            """;

    private void writeModel(Path dir, boolean dropOneTensor) throws Exception {
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("config.json"), CONFIG_JSON);
        Gemma3Config config = new Gemma3ConfigReader().read(CONFIG_JSON);
        Map<String, long[]> shapes = new LinkedHashMap<>(Gemma3TensorNameMapper.expectedShapes(config));
        if (dropOneTensor) {
            shapes.remove(Gemma3TensorNameMapper.qNorm(0));
        }
        writeSafeTensors(dir.resolve("model.safetensors"), shapes);
    }

    @Test
    void compilesAndReopensAndRunsNativeForward() throws Exception {
        writeModel(tempDir, false);
        Gemma3WdmlPackCompiler.CompileResult r = Gemma3WdmlPackCompiler.compile(
                tempDir, tempDir.resolve(Gemma3WdmlPackCompiler.DEFAULT_OUTPUT_NAME), false);
        assertTrue(r.written());
        assertTrue(r.runtimeLoadable(), "complete model must be runtime-loadable: missing=" + r.missing()
                + " shapeErrors=" + r.shapeErrors());
        assertTrue(Files.isRegularFile(r.output()));

        Gemma3RuntimePackage pkg = Gemma3RuntimePackage.open(r.output());
        assertTrue(pkg.runtimeLoadable());
        assertEquals(8, pkg.config().hiddenSize());
        assertEquals(Gemma3WdmlPackCompiler.RUNTIME_LOAD_MODE, pkg.runtimeLoadMode());

        Gemma3ReferenceWeights weights = pkg.loadReferenceWeights();
        Gemma3ReferenceForwardPass fp = new Gemma3ReferenceForwardPass(weights);
        float[] logits = fp.logitsForLastToken(new int[]{2, 5, 9});
        assertEquals(20, logits.length);
        for (float v : logits) {
            assertTrue(Float.isFinite(v));
        }
    }

    @Test
    void missingTensorIsNotRuntimeLoadable() throws Exception {
        writeModel(tempDir, true);
        Gemma3WdmlPackCompiler.CompileResult r = Gemma3WdmlPackCompiler.compile(
                tempDir, tempDir.resolve(Gemma3WdmlPackCompiler.DEFAULT_OUTPUT_NAME), false);
        assertFalse(r.runtimeLoadable());
        assertTrue(r.missing().contains(Gemma3TensorNameMapper.qNorm(0)));
    }

    private static void writeSafeTensors(Path file, Map<String, long[]> tensors) throws Exception {
        StringBuilder header = new StringBuilder("{");
        long offset = 0;
        boolean first = true;
        // First pass: header with offsets.
        Map<String, long[]> ranges = new LinkedHashMap<>();
        for (Map.Entry<String, long[]> e : tensors.entrySet()) {
            long n = 1;
            for (long d : e.getValue()) n *= d;
            long bytes = n * Float.BYTES;
            if (!first) header.append(',');
            first = false;
            header.append('"').append(e.getKey()).append("\":{\"dtype\":\"F32\",\"shape\":")
                    .append(shapeJson(e.getValue())).append(",\"data_offsets\":[")
                    .append(offset).append(',').append(offset + bytes).append("]}");
            ranges.put(e.getKey(), new long[]{n, offset});
            offset += bytes;
        }
        header.append('}');
        byte[] headerBytes = header.toString().getBytes(StandardCharsets.UTF_8);
        ByteBuffer out = ByteBuffer.allocate(Long.BYTES + headerBytes.length + (int) offset)
                .order(ByteOrder.LITTLE_ENDIAN);
        out.putLong(headerBytes.length);
        out.put(headerBytes);
        int seed = 0;
        for (Map.Entry<String, long[]> e : tensors.entrySet()) {
            long n = ranges.get(e.getKey())[0];
            for (long i = 0; i < n; i++) {
                out.putFloat(((seed++ % 7) - 3) * 0.01f);  // small finite values
            }
        }
        Files.write(file, out.array());
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
