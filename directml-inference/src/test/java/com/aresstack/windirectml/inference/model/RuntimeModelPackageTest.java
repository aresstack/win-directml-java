package com.aresstack.windirectml.inference.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RuntimeModelPackageTest {

    @TempDir
    Path tempDir;

    @Test
    void mapsPayloadTensorsWithoutModelSpecificCode() throws Exception {
        Path pack = tempDir.resolve("sample.wdmlpack");
        Map<String, Object> manifest = manifest(true);
        manifest.put("tensors", List.of(Map.of(
                "name", "encoder.layers.000.self.q.weight",
                "dataType", 10,
                "dims", List.of(2L, 2L),
                "storageKind", "INLINE",
                "byteLength", 4L,
                "payloadOffset", 0L,
                "payloadLength", 4L
        )));

        WdmlPackWriter.writeWithPayload(pack, manifest, List.of(
                new WdmlPackWriter.PayloadEntry("encoder.layers.000.self.q.weight", 0L, 4L,
                        channel -> channel.write(ByteBuffer.wrap(new byte[]{4, 3, 2, 1})))
        ), 4L);

        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(pack);
        Map<String, RuntimeTensor> tensors = modelPackage.mapPayloadTensors();
        RuntimeTensor tensor = tensors.get("encoder.layers.000.self.q.weight");

        assertTrue(modelPackage.payloadIncluded());
        assertEquals(1, tensors.size());
        assertEquals(4, tensor.rawByteLength());
        assertEquals(4, tensor.rawDataBuffer().get(0));
        assertEquals(1, modelPackage.buildTensorCatalog().size());
    }

    @Test
    void validatesSourceFingerprint() throws Exception {
        Path source = tempDir.resolve("model.safetensors");
        Files.write(source, new byte[]{1, 2, 3});
        SourceFingerprint fingerprint = SourceFingerprint.read(source);
        Path pack = tempDir.resolve("sample.wdmlpack");
        Map<String, Object> manifest = manifest(false);
        manifest.put("source", Map.of(
                "fileName", source.getFileName().toString(),
                "relativePath", source.getFileName().toString(),
                "sizeBytes", fingerprint.sizeBytes(),
                "lastModifiedMillis", fingerprint.lastModifiedMillis(),
                "fileKey", fingerprint.fileKey(),
                "fingerprint", fingerprint.value()
        ));
        manifest.put("tensors", List.of());
        WdmlPackWriter.writeManifestOnly(pack, manifest);

        RuntimeModelPackage modelPackage = RuntimeModelPackage.open(pack);

        assertFalse(modelPackage.payloadIncluded());
        assertEquals(source.toAbsolutePath().normalize(), modelPackage.resolveSourcePath(tempDir, "fallback.onnx"));
        modelPackage.validateSourceFingerprint(source);
    }

    private static Map<String, Object> manifest(boolean payloadIncluded) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("mode", payloadIncluded ? "payload" : "manifest-only");
        root.put("payloadIncluded", payloadIncluded);
        root.put("runtimeLoadable", true);
        root.put("runtimeLoadMode", payloadIncluded ? "native-payload" : "frontdoor");
        root.put("source", Map.of(
                "fileName", "model.onnx",
                "relativePath", "model.onnx",
                "fingerprint", "model.onnx|-1|-1|"
        ));
        return root;
    }
}
