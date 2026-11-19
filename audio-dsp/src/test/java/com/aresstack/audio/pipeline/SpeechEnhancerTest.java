package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.enhance.BackendAvailability;
import com.aresstack.audio.enhance.SpeechEnhancementBackends;
import com.aresstack.audio.profile.AudioBlockType;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Slice 13A: the general Speech Enhancer with the pure-Java backend and the not-installed RNNoise adapter. */
public class SpeechEnhancerTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void pureJavaBackendReducesNoise() {
        short[] input = noise(48000, 4000);
        AudioBuffer out = registry.createProcessor(AudioBlockType.SPEECH_ENHANCER)
                .process(new AudioBuffer(input.clone(), MONO),
                        registry.defaultDefinition(AudioBlockType.SPEECH_ENHANCER, "e")
                                .withParameter("backend", "PURE_JAVA_DSP")
                                .withParameter("strength", "1.0")
                                .withParameter("speechProtection", "false"),
                        new AudioProcessingContext());
        assertTrue("noise reduced: " + rms(out.getSamples()) + " vs " + rms(input),
                rms(out.getSamples()) < 0.9d * rms(input));
    }

    @Test
    public void missingRnnoiseBackendPassesAudioThrough() {
        assertEquals(BackendAvailability.NOT_INSTALLED,
                SpeechEnhancementBackends.availability("RNNOISE", MONO));
        short[] input = noise(8000, 4000);
        AudioBuffer out = registry.createProcessor(AudioBlockType.SPEECH_ENHANCER)
                .process(new AudioBuffer(input.clone(), MONO),
                        registry.defaultDefinition(AudioBlockType.SPEECH_ENHANCER, "e")
                                .withParameter("backend", "RNNOISE"),
                        new AudioProcessingContext());
        assertArrayEquals("missing backend must not change the audio", input, out.getSamples());
    }

    @Test
    public void registryExposesTheBuiltInBackends() {
        assertNotNull(SpeechEnhancementBackends.resolve("PURE_JAVA_DSP"));
        assertNotNull(SpeechEnhancementBackends.resolve("RNNOISE"));
        assertEquals(BackendAvailability.AVAILABLE,
                SpeechEnhancementBackends.availability("PURE_JAVA_DSP", MONO));
    }

    private static short[] noise(int n, int amp) {
        short[] out = new short[n];
        int state = 8080;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            out[i] = (short) (((state >> 16) % (2 * amp)) - amp);
        }
        return out;
    }

    private static double rms(short[] samples) {
        long sum = 0;
        int from = samples.length / 2;
        for (int i = from; i < samples.length; i++) {
            sum += (long) samples[i] * samples[i];
        }
        return Math.sqrt((double) sum / (samples.length - from));
    }
}
