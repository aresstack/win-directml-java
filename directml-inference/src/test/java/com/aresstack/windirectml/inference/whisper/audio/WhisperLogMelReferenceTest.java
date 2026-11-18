package com.aresstack.windirectml.inference.whisper.audio;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class WhisperLogMelReferenceTest {
    /**
     * Compare the Java frontend against torch.stft with OpenAI Whisper settings
     * and librosa.filters.mel(sr=16000, n_fft=400, n_mels=80).
     */
    @Test
    void matchesWhisperReferenceFeaturesForFortyFourCentihertzTone() {
        float[] samples = new float[16000];
        for (int sample = 0; sample < samples.length; sample++) {
            samples[sample] = (float) (0.6 * Math.sin(2.0 * Math.PI * 440.0 * sample / 16000.0));
        }
        float[][] features = new WhisperLogMelSpectrogram(80).compute(samples);

        assertEquals(80, features.length);
        assertEquals(3000, features[0].length);
        assertEquals(1.022869349f, features[0][0], 0.0001f);
        assertEquals(1.375831604f, features[10][0], 0.0001f);
        assertEquals(1.388328552f, features[10][30], 0.0001f);
        assertEquals(1.477794409f, features[11][30], 0.0001f);
        assertEquals(1.332512736f, features[11][100], 0.0001f);
        assertEquals(-0.522205591f, features[11][200], 0.0001f);
        assertEquals(-0.042099595f, features[79][0], 0.0001f);
        assertEquals(-0.192613959f, features[79][100], 0.0001f);
    }
}
