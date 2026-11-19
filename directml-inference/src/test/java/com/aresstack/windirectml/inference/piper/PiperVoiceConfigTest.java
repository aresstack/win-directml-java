package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class PiperVoiceConfigTest {

    @TempDir
    Path temp;

    @Test
    void readsVoiceSettingsAndProtectsPhonemeIdsFromMutation() throws Exception {
        PiperVoiceConfig voice = PiperVoiceConfig.load(writeConfig(validJson()));

        assertEquals(22050, voice.sampleRate());
        assertEquals(1, voice.speakerCount());
        assertEquals("de", voice.espeakVoice());
        assertEquals(0.5f, voice.noiseScale());
        assertEquals(1.0f, voice.lengthScale());
        assertEquals(0.333f, voice.noiseScaleW());
        assertEquals(6, voice.phonemeCount());
        int[] ids = voice.idsFor("a");
        ids[0] = 999;
        assertEquals(10, voice.idsFor("a")[0]);
    }

    @Test
    void rejectsMissingPiperMarkersAndUnknownPhonemeType() throws Exception {
        String invalid = validJson().replace("\"$\" : [2],", "");
        assertTrue(assertThrows(java.io.IOException.class,
                () -> PiperVoiceConfig.load(writeConfig(invalid))).getMessage()
                .contains("marker"));
        assertTrue(assertThrows(java.io.IOException.class,
                () -> PiperVoiceConfig.load(writeConfig(validJson().replace(
                        "\"phoneme_type\": \"espeak\"", "\"phoneme_type\": \"text\""))))
                .getMessage().contains("eSpeak"));
    }

    private Path writeConfig(String json) throws Exception {
        Path config = temp.resolve("voice-" + System.nanoTime() + ".onnx.json");
        Files.writeString(config, json);
        return config;
    }

    static String validJson() {
        return """
                {
                  "audio": {"sample_rate": 22050},
                  "espeak": {"voice": "de"},
                  "phoneme_type": "espeak",
                  "num_speakers": 1,
                  "inference": {"noise_scale": 0.5, "length_scale": 1.0, "noise_w": 0.333},
                  "phoneme_id_map": {
                    "^": [1], "_": [0], "$" : [2],
                    "a": [10], "ː": [11], " ": [3]
                  }
                }
                """;
    }
}
