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

/** Slice 13B: the DeepFilterNet backend is registered, reports not-installed and passes audio through. */
public class DeepFilterNetBackendTest {

    private static final PcmAudioFormat MONO = new PcmAudioFormat(16000, 1, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void deepFilterNetIsRegisteredButNotInstalled() {
        assertNotNull(SpeechEnhancementBackends.resolve("DEEPFILTERNET"));
        assertEquals(BackendAvailability.NOT_INSTALLED,
                SpeechEnhancementBackends.availability("DEEPFILTERNET", MONO));
    }

    @Test
    public void missingDeepFilterNetPassesAudioThrough() {
        short[] input = new short[2000];
        for (int i = 0; i < input.length; i++) {
            input[i] = (short) (i % 200 - 100);
        }
        AudioBuffer out = registry.createProcessor(AudioBlockType.SPEECH_ENHANCER)
                .process(new AudioBuffer(input.clone(), MONO),
                        registry.defaultDefinition(AudioBlockType.SPEECH_ENHANCER, "e")
                                .withParameter("backend", "DEEPFILTERNET")
                                .withParameter("modelId", "deepfilternet3"),
                        new AudioProcessingContext());
        assertArrayEquals(input, out.getSamples());
    }
}
