package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class PiperOnnxModelSourceTest {

    @Test
    @EnabledIfSystemProperty(named = "piper.realModel", matches = "true")
    void importsTheRealGermanPiperVoiceWithoutOnnxRuntime() throws Exception {
        Path model = Path.of(System.getProperty("piper.modelFile",
                "../model/piper-de_DE-thorsten-medium/de_DE-thorsten-medium.onnx"));
        assertTrue(Files.isRegularFile(model), "Download the opted-in Piper test model first");

        PiperModelImport imported = new PiperOnnxModelSource(model).load();

        assertEquals(22050, imported.voice().sampleRate());
        assertEquals("de", imported.voice().espeakVoice());
        assertTrue(imported.graph().inputNames().containsAll(
                Set.of("input", "input_lengths", "scales")));
        assertTrue(imported.graph().outputNames().contains("output"));
        assertTrue(imported.operators().getOrDefault("Conv", 0L) > 0);
        assertTrue(imported.operators().getOrDefault("ConvTranspose", 0L) > 0);
        assertTrue(imported.nodeCount() > 2000);
        assertFalse(imported.graph().initializers().isEmpty());
        System.out.println("Piper VITS parsed: nodes=" + imported.nodeCount()
                + " operators=" + imported.operators());
    }
}
