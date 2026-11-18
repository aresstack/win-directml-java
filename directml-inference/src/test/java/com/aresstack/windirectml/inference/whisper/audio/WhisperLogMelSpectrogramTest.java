package com.aresstack.windirectml.inference.whisper.audio;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WhisperLogMelSpectrogramTest {
    @Test
    void silentAudioProducesWhisperLogFloorForBothMelDimensions() {
        for (int melBands : new int[]{80, 128}) {
            float[][] spectrum = new WhisperLogMelSpectrogram(melBands).compute(new float[0]);
            assertEquals(melBands, spectrum.length);
            for (float[] band : spectrum) {
                assertEquals(3000, band.length);
                assertEquals(-1.5f, band[0], 0.00001f);
                assertEquals(-1.5f, band[1000], 0.00001f);
                assertEquals(-1.5f, band[2999], 0.00001f);
            }
        }
    }

    @Test
    void detectsFourHundredFortyHertzToneInExpectedMelRange() {
        float[] samples = new float[16000];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (float) (0.6 * Math.sin(2 * Math.PI * 440.0 * i / 16000));
        }
        float[][] spectrum = new WhisperLogMelSpectrogram(80).compute(samples);
        int brightestBand = 0;
        for (int band = 1; band < 80; band++) {
            if (spectrum[band][30] > spectrum[brightestBand][30]) {
                brightestBand = band;
            }
        }
        assertTrue(brightestBand >= 10 && brightestBand <= 18, "440 Hz Mel band: " + brightestBand);
        assertTrue(spectrum[brightestBand][30] > spectrum[brightestBand][200]);
        assertTrue(Float.isFinite(spectrum[brightestBand][30]));
    }

    @Test
    void acceptsPartialChunkWithOffsetAndRejectsInvalidOffset() {
        float[] samples = new float[300];
        WhisperLogMelSpectrogram frontend = new WhisperLogMelSpectrogram(80);
        assertEquals(3000, frontend.compute(samples, 150)[0].length);
        assertThrows(IllegalArgumentException.class, () -> frontend.compute(samples, 301));
    }

    @Test
    void rejectsUnsupportedMelDimensions() {
        assertThrows(IllegalArgumentException.class, () -> new WhisperLogMelSpectrogram(40));
    }
}
