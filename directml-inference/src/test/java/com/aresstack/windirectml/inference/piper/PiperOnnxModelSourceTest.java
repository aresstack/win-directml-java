package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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

        PiperSynthesisInput tokens = PiperSynthesisInput.of(imported.voice(), "a", 0);
        PiperNativeGraphPrefixRunner runner = new PiperNativeGraphPrefixRunner(42L);
        PiperNativeGraphPrefixRunner.PrefixResult scalar = runner.execute(imported, tokens, 1200);
        assertEquals(1200, scalar.executedNodes());
        assertArrayEquals(new int[0], scalar.output().shape());
        assertEquals(1, scalar.output().length());

        PiperNativeGraphPrefixRunner.PrefixResult prefix = runner.execute(imported, tokens, imported.nodeCount());
        assertEquals(imported.nodeCount(), prefix.executedNodes());
        assertEquals(imported.graph().nodes().get(prefix.executedNodes() - 1).opType(),
                prefix.lastOperator());
        assertEquals(PiperTensor.FLOAT32, prefix.output().dataType());
        assertTrue(prefix.output().rank() >= 3);
        float[] audio = prefix.output().floatData();
        assertTrue(audio.length > 100, "Piper must emit an actual audio waveform");
        boolean audibleSamples = false;
        for (float sample : audio) {
            assertTrue(Float.isFinite(sample), "The output must contain only finite PCM values");
            assertTrue(sample >= -1.001f && sample <= 1.001f,
                    "The final Tanh must bound output samples");
            if (Math.abs(sample) > 0.0001f) {
                audibleSamples = true;
            }
        }
        assertTrue(audibleSamples, "The waveform must not be silent");

        System.out.println("Piper VITS parsed: nodes=" + imported.nodeCount()
                + " operators=" + imported.operators());
    }
}
