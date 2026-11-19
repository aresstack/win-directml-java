package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class PiperReferenceSynthesizerTest {

    @Test
    @EnabledIfSystemProperty(named = "piper.realModel", matches = "true")
    void createsARealGermanVoiceWaveformAndExportsWav() throws Exception {
        Path model = Path.of(System.getProperty("piper.modelFile",
                "../model/piper-de_DE-thorsten-medium/de_DE-thorsten-medium.onnx"));
        assertTrue(Files.isRegularFile(model), "Install the local opt-in Piper voice first");
        PiperModelImport imported = new PiperOnnxModelSource(model).load();
        PiperReferenceSynthesizer synthesizer = new PiperReferenceSynthesizer(imported);

        PiperAudio audio = synthesizer.synthesizePhonemes("hˈaloː");
        PiperAudio fromText = synthesizer.synthesizeText("Hallo", (text, language) -> {
            assertEquals("Hallo", text);
            assertEquals("de", language);
            return "hˈaloː";
        });
        org.junit.jupiter.api.Assertions.assertArrayEquals(
                audio.samples(), fromText.samples(), 0f);
        Path destination = Path.of("build", "piper-demo", "thorsten-hallo.wav");
        new PiperWavWriter().write(audio, destination);

        assertEquals(22_050, audio.sampleRate());
        assertTrue(audio.sampleCount() > 2000);
        assertTrue(audio.durationSeconds() > 0.05);
        assertTrue(Files.size(destination) > 4000);
        System.out.println("Piper WAV: " + destination.toAbsolutePath()
                + " | samples=" + audio.sampleCount()
                + " | durationSeconds=" + audio.durationSeconds());
    }
}
