package com.aresstack.windirectml.inference.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WdmlPackReaderTest {

    @TempDir
    Path tempDir;

    @Test
    void opensPackageAsDescriptorAndMapsRuntimeTensorCatalog() throws Exception {
        Path pack = tempDir.resolve("runtime.wdmlpack");
        Map<String, Object> manifest = manifest();
        manifest.put("tensors", List.of(Map.of(
                "name", "decoder.layers.000.self_attn.q_proj.weight",
                "dataType", 10,
                "dims", List.of(2L, 2L),
                "storageKind", "INLINE",
                "byteLength", 4L,
                "payloadOffset", 0L,
                "payloadLength", 4L
        )));

        WdmlPackWriter.writeWithPayload(pack, manifest, List.of(
                new WdmlPackWriter.PayloadEntry("decoder.layers.000.self_attn.q_proj.weight", 0L, 4L,
                        channel -> channel.write(ByteBuffer.wrap(new byte[]{11, 12, 13, 14})))
        ), 4L);

        RuntimeModelPackage modelPackage = WdmlPackReader.open(pack);
        RuntimeTensorCatalog tensors = modelPackage.runtimeTensorCatalog();

        assertTrue(modelPackage.payloadIncluded());
        assertEquals(1, tensors.size());
        assertEquals(4, tensors.payloadBytes());
        assertEquals(11, tensors.get("decoder.layers.000.self_attn.q_proj.weight").rawDataBuffer().get(0));
        assertEquals(1, tensors.toSourceTensorCatalog().size());
        assertEquals(1, tensors.toTensorCatalog().size());
    }

    private static Map<String, Object> manifest() {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("format", "wdmlpack");
        root.put("version", WdmlPackWriter.VERSION);
        root.put("mode", "payload");
        root.put("payloadIncluded", true);
        root.put("runtimeLoadable", true);
        root.put("runtimeLoadMode", "native-payload");
        root.put("source", Map.of(
                "fileName", "model.safetensors",
                "relativePath", "model.safetensors",
                "fingerprint", "model.safetensors|-1|-1|"
        ));
        return root;
    }
}
