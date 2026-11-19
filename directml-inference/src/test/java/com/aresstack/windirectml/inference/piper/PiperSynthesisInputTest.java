package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class PiperSynthesisInputTest {

    @TempDir
    Path folder;

    @Test
    void suppliesTypedVitsInputsInPiperOrderWithoutExposingMutableArrays() throws Exception {
        Path config = folder.resolve("voice.onnx.json");
        Files.writeString(config, PiperVoiceConfigTest.validJson());
        PiperVoiceConfig voice = PiperVoiceConfig.load(config);

        PiperSynthesisInput input = PiperSynthesisInput.of(voice, "aː", 0);

        assertArrayEquals(new long[]{1, 0, 10, 0, 11, 0, 2}, input.phonemeIds());
        assertArrayEquals(new long[]{7}, input.phonemeLengths());
        assertArrayEquals(new long[]{1, 7}, input.inputShape());
        assertArrayEquals(new long[]{3}, input.scalesShape());
        assertArrayEquals(new float[]{0.5f, 1f, 0.333f}, input.scales());
        assertEquals(0, input.speakerId());

        long[] mutatedIds = input.phonemeIds();
        mutatedIds[0] = 1234;
        assertEquals(1, input.phonemeIds()[0]);
    }

    @Test
    void rejectsSpeakerIdsOutsideTheVoiceRange() throws Exception {
        Path config = folder.resolve("voice.onnx.json");
        Files.writeString(config, PiperVoiceConfigTest.validJson());
        PiperVoiceConfig voice = PiperVoiceConfig.load(config);

        assertThrows(IllegalArgumentException.class,
                () -> PiperSynthesisInput.of(voice, "a", 1));
    }
}
