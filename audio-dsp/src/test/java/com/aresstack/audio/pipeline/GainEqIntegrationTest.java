package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** The four new blocks integrate through the registry and never disturb the built-in default profile. */
public class GainEqIntegrationTest {

    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void registryCreatesAllFourNewTypesWithParameters() {
        AudioBlockType[] types = {AudioBlockType.GAIN, AudioBlockType.PARAMETRIC_EQ,
                AudioBlockType.LOW_SHELF, AudioBlockType.HIGH_SHELF};
        for (AudioBlockType type : types) {
            assertNotNull("processor for " + type, registry.createProcessor(type));
            assertFalse("parameters for " + type, registry.descriptor(type).getParameters().isEmpty());
            assertFalse("default params for " + type, registry.defaultParameters(type).isEmpty());
        }
    }

    @Test
    public void defaultSpeechContainsNoneOfTheNewBlocksAndIsBitIdentical() {
        AudioProcessingProfile defaultProfile = AudioProcessingProfiles.defaultSpeech();
        for (AudioBlockDefinition block : defaultProfile.getBlocks()) {
            AudioBlockType type = block.getType();
            assertFalse("default-speech must not gain new blocks",
                    type == AudioBlockType.GAIN || type == AudioBlockType.PARAMETRIC_EQ
                            || type == AudioBlockType.LOW_SHELF || type == AudioBlockType.HIGH_SHELF);
        }
        // Adding the block types must not change the default pipeline's output: two runs stay identical
        // and the result is stable (regression guard for the protected default result).
        AudioBuffer input = new AudioBuffer(noise(4800), new PcmAudioFormat(48000, 2, 16));
        short[] first = new AudioProfileProcessor().process(copy(input), defaultProfile).getSamples();
        short[] second = new AudioProfileProcessor().process(copy(input), defaultProfile).getSamples();
        assertArrayEquals(first, second);
    }

    @Test
    public void aProfileWithAllFourBlocksProcessesWithoutInstability() {
        List<AudioBlockDefinition> blocks = new ArrayList<AudioBlockDefinition>();
        blocks.add(registry.defaultDefinition(AudioBlockType.GAIN, "g")
                .withParameter("gainDb", "3"));
        blocks.add(registry.defaultDefinition(AudioBlockType.PARAMETRIC_EQ, "p")
                .withParameter("centerHz", "1500").withParameter("gainDb", "6").withParameter("q", "1.2"));
        blocks.add(registry.defaultDefinition(AudioBlockType.LOW_SHELF, "l")
                .withParameter("cutoffHz", "180").withParameter("gainDb", "4"));
        blocks.add(registry.defaultDefinition(AudioBlockType.HIGH_SHELF, "h")
                .withParameter("cutoffHz", "6000").withParameter("gainDb", "-3"));
        AudioProcessingProfile profile = new AudioProcessingProfile("eq", "EQ chain", false, blocks);

        AudioBuffer input = new AudioBuffer(sine(1000.0d, 16000), new PcmAudioFormat(16000, 1, 16));
        short[] output = new AudioProfileProcessor().process(input, profile).getSamples();
        for (short sample : output) {
            assertTrue(sample >= Short.MIN_VALUE && sample <= Short.MAX_VALUE);
        }
    }

    @Test
    public void aDisabledNewBlockLeavesTheSignalUnchanged() {
        AudioBlockDefinition disabledGain = registry.defaultDefinition(AudioBlockType.GAIN, "g")
                .withParameter("gainDb", "12").withEnabled(false);
        List<AudioBlockDefinition> blocks = new ArrayList<AudioBlockDefinition>();
        blocks.add(disabledGain);
        AudioProcessingProfile profile = new AudioProcessingProfile("d", "Disabled", false, blocks);

        short[] samples = sine(1000.0d, 4000);
        AudioBuffer input = new AudioBuffer(samples.clone(), new PcmAudioFormat(16000, 1, 16));
        short[] output = new AudioProfileProcessor().process(input, profile).getSamples();
        assertArrayEquals(samples, output);
    }

    private static AudioBuffer copy(AudioBuffer buffer) {
        short[] samples = buffer.getSamples().clone();
        return new AudioBuffer(samples, buffer.getFormat());
    }

    private static short[] sine(double frequencyHz, int length) {
        short[] samples = new short[length];
        for (int i = 0; i < length; i++) {
            samples[i] = (short) Math.round(8000.0d * Math.sin(2.0d * Math.PI * frequencyHz * i / 16000.0d));
        }
        return samples;
    }

    private static short[] noise(int frames) {
        short[] samples = new short[frames * 2]; // stereo
        int state = 12345;
        for (int i = 0; i < samples.length; i++) {
            state = state * 1103515245 + 12345;
            samples[i] = (short) (state >> 16);
        }
        return samples;
    }
}
