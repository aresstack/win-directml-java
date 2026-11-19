package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class PiperPhonemeIdEncoderTest {

    @TempDir
    Path temp;

    @Test
    void placesBeginningSeparatorAndEndingAroundUnicodeIpaPhonemes() throws Exception {
        Path json = temp.resolve("test.onnx.json");
        Files.writeString(json, PiperVoiceConfigTest.validJson());
        PiperPhonemeIdEncoder encoder = new PiperPhonemeIdEncoder(PiperVoiceConfig.load(json));

        assertArrayEquals(new long[]{1, 0, 10, 0, 11, 0, 2}, encoder.encode("aː"));
    }

    @Test
    void rejectsUnknownPhonemesInsteadOfGeneratingWrongSpeech() throws Exception {
        Path json = temp.resolve("test.onnx.json");
        Files.writeString(json, PiperVoiceConfigTest.validJson());
        PiperPhonemeIdEncoder encoder = new PiperPhonemeIdEncoder(PiperVoiceConfig.load(json));

        assertThrows(IllegalArgumentException.class, () -> encoder.encode("X"));
        assertThrows(IllegalArgumentException.class, () -> encoder.encode(" "));
    }
}
