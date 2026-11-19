package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.enhance.BackendAvailability;
import com.aresstack.audio.enhance.SpeechEnhancementBackends;
import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Slice 13C: the combined pure-Java denoise+dereverb backend is available and reduces noise. */
public class CombinedEnhancerTest {

    private static final PcmAudioFormat MONO = new PcmAudioFormat(16000, 1, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void combinedBackendIsAvailable() {
        assertNotNull(SpeechEnhancementBackends.resolve("COMBINED_DENOISE_DEREVERB"));
        assertEquals(BackendAvailability.AVAILABLE,
                SpeechEnhancementBackends.availability("COMBINED_DENOISE_DEREVERB", MONO));
    }

    @Test
    public void combinedBackendReducesNoise() {
        short[] input = noise(48000, 4000);
        AudioBuffer out = registry.createProcessor(AudioBlockType.SPEECH_ENHANCER)
                .process(new AudioBuffer(input.clone(), MONO),
                        registry.defaultDefinition(AudioBlockType.SPEECH_ENHANCER, "e")
                                .withParameter("backend", "COMBINED_DENOISE_DEREVERB")
                                .withParameter("strength", "1.0")
                                .withParameter("speechProtection", "false"),
                        new AudioProcessingContext());
        assertTrue("combined enhancer reduces noise: " + rms(out.getSamples()) + " vs " + rms(input),
                rms(out.getSamples()) < 0.95d * rms(input));
    }

    private static short[] noise(int n, int amp) {
        short[] out = new short[n];
        int state = 5150;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            out[i] = (short) (((state >> 16) % (2 * amp)) - amp);
        }
        return out;
    }

    private static double rms(short[] s) {
        long sum = 0;
        int from = s.length / 2;
        for (int i = from; i < s.length; i++) {
            sum += (long) s[i] * s[i];
        }
        return Math.sqrt((double) sum / (s.length - from));
    }
}
