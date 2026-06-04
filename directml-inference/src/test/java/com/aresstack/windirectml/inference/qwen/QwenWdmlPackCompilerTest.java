package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.TensorCatalog;
import com.aresstack.windirectml.inference.model.TensorEntry;
import com.aresstack.windirectml.inference.model.TensorStorageKind;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxGraph;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxNode;
import com.aresstack.windirectml.windows.OnnxModelReader.OnnxTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class QwenWdmlPackCompilerTest {

    @TempDir
    Path tempDir;

    @Test
    void manifestContainsQwenMetadataAndTensorDirectory() throws Exception {
        Path onnx = tempDir.resolve("model_q4f16.onnx");
        Files.write(onnx, new byte[]{1, 2, 3, 4});

        Map<String, OnnxTensor> initializers = new LinkedHashMap<>();
        initializers.put("model.embed_tokens.weight", new OnnxTensor(
                "model.embed_tokens.weight", new long[]{151936, 896}, 10, new float[0], new byte[0]));
        OnnxGraph graph = new OnnxGraph("main_graph",
                List.of(new OnnxNode("MatMulNBits", List.of("x", "w", "s"), List.of("y"), Map.of())),
                initializers, List.of("input_ids"), List.of("logits"));
        TensorCatalog catalog = new TensorCatalog(List.of(
                new TensorEntry("model.embed_tokens.weight", 10, new long[]{151936, 896},
                        TensorStorageKind.INLINE, 272269312L)));
        QwenModelImport imported = new QwenModelImport("onnx", onnx, graph,
                Map.of(), initializers, catalog);
        Qwen2Config config = new Qwen2Config(896, 14, 24, 2, 151936,
                32768, 4864, 1e-6f, 1_000_000f, true);

        Map<String, Object> manifest = QwenWdmlPackCompiler.buildManifest(
                imported, config, tempDir, "model_q4f16.onnx");

        assertEquals("wdmlpack", manifest.get("format"));
        assertEquals("manifest-only", manifest.get("mode"));
        assertEquals(false, manifest.get("payloadIncluded"));

        @SuppressWarnings("unchecked")
        Map<String, Object> model = (Map<String, Object>) manifest.get("model");
        assertEquals("qwen2", model.get("architecture"));
        assertEquals(896, model.get("hiddenSize"));
        assertEquals(151936, model.get("vocabSize"));


        @SuppressWarnings("unchecked")
        Map<String, Object> cache = (Map<String, Object>) manifest.get("cache");
        assertEquals(QwenWdmlPackCompiler.CACHE_SCHEMA, cache.get("schema"));
        assertEquals(QwenWdmlPackCompiler.COMPILER_VERSION, cache.get("compilerVersion"));

        @SuppressWarnings("unchecked")
        Map<String, Object> source = (Map<String, Object>) manifest.get("source");
        assertEquals("onnx", source.get("format"));
        assertEquals(1L, source.get("matMulNBitsNodes"));
        assertTrue(((String) source.get("fingerprint")).contains("model_q4f16.onnx"));

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> tensors = (List<Map<String, Object>>) manifest.get("tensors");
        assertEquals(1, tensors.size());
        assertEquals("model.embed_tokens.weight", tensors.get(0).get("name"));
        assertEquals(-1L, tensors.get(0).get("payloadOffset"));
    }
}
