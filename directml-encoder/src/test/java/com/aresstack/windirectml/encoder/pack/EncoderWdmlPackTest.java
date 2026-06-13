package com.aresstack.windirectml.encoder.pack;

import com.aresstack.windirectml.encoder.safetensors.SafetensorsReader;
import com.aresstack.windirectml.inference.artifact.ModelArtifactStatus;
import com.aresstack.windirectml.inference.artifact.PackageState;
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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * ER-WDML: the encoder/reranker wdmlpack compiler + reader round-trips SafeTensors weights bit-exactly
 * (parity by construction), and the lifecycle treats encoders as a real download -> convert -> run flow.
 */
class EncoderWdmlPackTest {

    @TempDir
    Path tempDir;

    @Test
    void compileThenReadBackIsBitExact() throws Exception {
        Map<String, float[]> tensors = new LinkedHashMap<>();
        tensors.put("embeddings.word_embeddings.weight", new float[]{0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.6f});
        tensors.put("encoder.layer.0.output.LayerNorm.bias", new float[]{1.0f, 2.0f, 3.0f});
        Path safetensors = tempDir.resolve("model.safetensors");
        writeF32SafeTensors(safetensors, tensors);

        Path pkg = EncoderWdmlPack.compile(tempDir, tempDir.resolve(EncoderWdmlPack.ENCODER_PACKAGE_FILE),
                EncoderWdmlPack.FAMILY_ENCODER);
        assertTrue(Files.isRegularFile(pkg));
        assertTrue(RuntimeModelPackage.open(pkg).runtimeLoadable(), "encoder package must be runtime-loadable");

        try (SafetensorsReader fromPackage = EncoderWdmlPack.openWeightsReader(pkg);
             SafetensorsReader fromFile = SafetensorsReader.open(safetensors)) {
            for (String name : tensors.keySet()) {
                assertArrayEquals(fromFile.readFloat32(name), fromPackage.readFloat32(name),
                        "weights read from the package must equal the original SafeTensors: " + name);
            }
        }
    }

    @Test
    void embeddingLifecycleConvertsAndBecomesReadyWithoutWritingDuringInspect() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), "{}");
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        Map<String, float[]> tensors = new LinkedHashMap<>();
        tensors.put("embeddings.word_embeddings.weight", new float[]{0.1f, 0.2f});
        writeF32SafeTensors(tempDir.resolve("model.safetensors"), tensors);

        EncoderPackageLifecycle lifecycle = EncoderPackageLifecycle.embedding();

        ModelArtifactStatus before = lifecycle.inspect(tempDir);
        assertEquals(PackageState.PACKAGE_MISSING, before.packageState());
        assertFalse(before.ready());
        assertThrows(IllegalStateException.class, () -> lifecycle.validateOrThrowBeforeInference(tempDir));
        assertFalse(Files.exists(tempDir.resolve(EncoderWdmlPack.ENCODER_PACKAGE_FILE)),
                "inspect/validate must not write a package");

        assertTrue(lifecycle.convert(tempDir, false).ok());

        ModelArtifactStatus after = lifecycle.inspect(tempDir);
        assertEquals(PackageState.PACKAGE_VALID, after.packageState());
        assertTrue(after.ready());
        lifecycle.validateOrThrowBeforeInference(tempDir); // must not throw
    }

    @Test
    void rerankerLifecycleUsesRerankerPackageFile() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), "{}");
        Files.writeString(tempDir.resolve("tokenizer.json"), "{}");
        Map<String, float[]> tensors = new LinkedHashMap<>();
        tensors.put("classifier.weight", new float[]{0.5f, -0.5f});
        writeF32SafeTensors(tempDir.resolve("model.safetensors"), tensors);

        EncoderPackageLifecycle lifecycle = EncoderPackageLifecycle.reranker();
        assertTrue(lifecycle.convert(tempDir, false).ok());
        assertTrue(Files.isRegularFile(tempDir.resolve(EncoderWdmlPack.RERANKER_PACKAGE_FILE)));
        assertTrue(lifecycle.inspect(tempDir).ready());
    }

    // --- minimal F32 .safetensors writer -----------------------------------------------------

    private static void writeF32SafeTensors(Path file, Map<String, float[]> tensors) throws Exception {
        StringBuilder header = new StringBuilder("{");
        int offset = 0;
        boolean first = true;
        for (Map.Entry<String, float[]> e : tensors.entrySet()) {
            int bytes = e.getValue().length * Float.BYTES;
            if (!first) header.append(',');
            first = false;
            header.append('"').append(e.getKey()).append("\":{\"dtype\":\"F32\",\"shape\":[")
                    .append(e.getValue().length).append("],\"data_offsets\":[")
                    .append(offset).append(',').append(offset + bytes).append("]}");
            offset += bytes;
        }
        header.append('}');
        byte[] headerBytes = header.toString().getBytes(StandardCharsets.UTF_8);
        ByteBuffer out = ByteBuffer.allocate(Long.BYTES + headerBytes.length + offset).order(ByteOrder.LITTLE_ENDIAN);
        out.putLong(headerBytes.length);
        out.put(headerBytes);
        for (float[] data : tensors.values()) {
            for (float v : data) {
                out.putFloat(v);
            }
        }
        Files.write(file, out.array());
    }
}
