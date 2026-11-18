package com.aresstack.windirectml.inference.whisper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.Assumptions;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class WhisperCheckpointTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void importsFloat16IncludingSubnormalValues() throws Exception {
        Path checkpointFile = temporaryDirectory.resolve("tiny.safetensors");
        String header = "{\"model.encoder.test.weight\":{\"dtype\":\"F16\","
                + "\"shape\":[4],\"data_offsets\":[0,8]}}";
        byte[] json = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer file = ByteBuffer.allocate(8 + json.length + 8).order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(json.length);
        file.put(json);
        file.putShort((short) 0x3c00);
        file.putShort((short) 0xbc00);
        file.putShort((short) 0x0001);
        file.putShort((short) 0x7bff);
        Files.write(checkpointFile, file.array());

        WhisperCheckpoint checkpoint = WhisperCheckpoint.open(checkpointFile);
        WhisperCheckpoint.Tensor tensor = checkpoint.require("model.encoder.test.weight");

        assertEquals(1, checkpoint.tensorCount());
        tensor.requireShape(4);
        assertEquals(1.0f, tensor.values()[0], 0.00001f);
        assertEquals(-1.0f, tensor.values()[1], 0.00001f);
        assertEquals(0x1.0p-24f, tensor.values()[2], 1.0e-12f);
        assertEquals(65504.0f, tensor.values()[3], 1.0e-5f);
        assertThrows(java.io.IOException.class, () -> checkpoint.require("nonexistent.weight"));
    }

    @Test
    void readsRealMultilingualWhisperTinyCheckpointIfInstalled() throws Exception {
        Path modelDir = Path.of("..", "model", "whisper-tiny");
        Assumptions.assumeTrue(Files.isRegularFile(modelDir.resolve("model.safetensors")));
        WhisperModelConfig settings = WhisperModelConfig.load(modelDir.resolve("config.json"));
        WhisperCheckpoint checkpoint = WhisperCheckpoint.open(modelDir.resolve("model.safetensors"));

        assertEquals(80, settings.melBands());
        assertEquals(384, settings.modelWidth());
        assertEquals(4, settings.encoderLayers());
        assertEquals(4, settings.decoderLayers());
        assertEquals(6, settings.encoderHeads());
        assertEquals(1500, settings.audioContext());
        assertTrue(checkpoint.tensorCount() > 100);
        checkpoint.require("model.encoder.conv1.weight").requireShape(384, 80, 3);
        checkpoint.require("model.encoder.conv1.bias").requireShape(384);
        checkpoint.require("model.decoder.embed_positions.weight").requireShape(448, 384);
    }
}
