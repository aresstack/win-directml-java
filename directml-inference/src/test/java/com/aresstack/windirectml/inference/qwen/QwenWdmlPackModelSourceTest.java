package com.aresstack.windirectml.inference.qwen;

import com.aresstack.windirectml.inference.model.WdmlPackWriter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class QwenWdmlPackModelSourceTest {

    @TempDir
    Path tempDir;

    @Test
    void rejectsPackageWhenSourceOnnxIsMissingBeforeDelegatingToOnnxParser() throws Exception {
        Path pack = tempDir.resolve("model_q4f16.wdmlpack");
        WdmlPackWriter.writeManifestOnly(pack, validManifest("model_q4f16.onnx", 1234L));

        QwenWdmlPackModelSource source = new QwenWdmlPackModelSource(
                tempDir, pack, "model_q4f16.onnx", config());

        IOException ex = assertThrows(IOException.class, source::load);
        assertTrue(ex.getMessage().contains("source ONNX is missing"));
    }

    @Test
    void rejectsPackageWhenConfigDoesNotMatchManifest() throws Exception {
        Path onnx = tempDir.resolve("model_q4f16.onnx");
        Files.write(onnx, new byte[]{1, 2, 3});
        Path pack = tempDir.resolve("model_q4f16.wdmlpack");
        Map<String, Object> manifest = validManifest("model_q4f16.onnx", Files.size(onnx));
        @SuppressWarnings("unchecked")
        Map<String, Object> model = (Map<String, Object>) manifest.get("model");
        model.put("hiddenSize", 999);
        WdmlPackWriter.writeManifestOnly(pack, manifest);

        QwenWdmlPackModelSource source = new QwenWdmlPackModelSource(
                tempDir, pack, "model_q4f16.onnx", config());

        IOException ex = assertThrows(IOException.class, source::load);
        assertTrue(ex.getMessage().contains("hiddenSize"));
    }

    @Test
    void loadsPayloadPackageWithoutSourceOnnx() throws Exception {
        Path pack = tempDir.resolve("model_q4f16.wdmlpack");
        Map<String, Object> manifest = validManifest("model_q4f16.onnx", -1L);
        manifest.put("mode", "payload");
        manifest.put("payloadIncluded", true);
        manifest.put("runtimeLoadMode", "wdmlpack-native-payload");

        Map<String, Object> runtimeGraph = new LinkedHashMap<>();
        runtimeGraph.put("nodes", java.util.List.of(Map.of(
                "opType", "MatMulNBits",
                "inputs", java.util.List.of("x", "tiny.weight_Q4", "tiny.scales"),
                "outputs", java.util.List.of("tiny.out")
        )));
        manifest.put("runtimeGraph", runtimeGraph);
        manifest.put("tensors", java.util.List.of(Map.of(
                "name", "tiny.weight_Q4",
                "dataType", 2,
                "dims", java.util.List.of(1L, 1L, 4L),
                "storageKind", "INLINE",
                "byteLength", 4L,
                "payloadOffset", 0L,
                "payloadLength", 4L
        )));

        WdmlPackWriter.writeWithPayload(pack, manifest, java.util.List.of(
                new WdmlPackWriter.PayloadEntry("tiny.weight_Q4", 0L, 4L,
                        channel -> channel.write(java.nio.ByteBuffer.wrap(new byte[]{9, 8, 7, 6})))
        ), 4L);

        QwenWdmlPackModelSource source = new QwenWdmlPackModelSource(
                tempDir, pack, "model_q4f16.onnx", config());

        QwenModelImport imported = source.load();
        assertEquals("wdmlpack-payload", imported.sourceFormat());
        assertEquals(pack.toAbsolutePath().normalize(), imported.modelPath());
        assertEquals(1, imported.graph().nodes().size());
        assertTrue(imported.externalRefs().isEmpty());
        assertEquals(4, imported.inlineTensors().get("tiny.weight_Q4").rawByteLength());
        assertEquals(9, imported.inlineTensors().get("tiny.weight_Q4").rawDataBuffer().get(0));
    }

    private static Map<String, Object> validManifest(String sourceFileName, long sizeBytes) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("mode", "manifest-only");
        root.put("payloadIncluded", false);
        root.put("runtimeLoadable", true);
        root.put("runtimeLoadMode", "wdmlpack-frontdoor-onnx-payload");

        Map<String, Object> source = new LinkedHashMap<>();
        source.put("format", "onnx");
        source.put("fileName", sourceFileName);
        source.put("relativePath", sourceFileName);
        source.put("sizeBytes", sizeBytes);
        root.put("source", source);

        Map<String, Object> model = new LinkedHashMap<>();
        model.put("architecture", "qwen2");
        model.put("hiddenSize", 896);
        model.put("numHiddenLayers", 24);
        model.put("numAttentionHeads", 14);
        model.put("numKeyValueHeads", 2);
        model.put("headDim", 64);
        model.put("vocabSize", 151936);
        model.put("intermediateSize", 4864);
        root.put("model", model);
        return root;
    }

    private static Qwen2Config config() {
        return new Qwen2Config(896, 14, 24, 2, 151936,
                32768, 4864, 1e-6f, 1_000_000f, true);
    }
}
