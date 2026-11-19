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

/** The four Slice-4C cleanup blocks integrate through the registry and never disturb the default profile. */
public class SpeechCleanupIntegrationTest {

    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void registryCreatesAllFourCleanupBlocks() {
        for (AudioBlockType type : new AudioBlockType[]{AudioBlockType.DE_ESSER,
                AudioBlockType.ADAPTIVE_HUM_REMOVAL, AudioBlockType.PLOSIVE_REDUCTION,
                AudioBlockType.BREATH_REDUCTION}) {
            assertNotNull(registry.createProcessor(type));
            assertFalse(registry.descriptor(type).getParameters().isEmpty());
        }
    }

    @Test
    public void defaultSpeechHasNoneOfThemAndStaysDeterministic() {
        AudioProcessingProfile def = AudioProcessingProfiles.defaultSpeech();
        for (AudioBlockDefinition block : def.getBlocks()) {
            AudioBlockType t = block.getType();
            assertFalse(t == AudioBlockType.DE_ESSER || t == AudioBlockType.ADAPTIVE_HUM_REMOVAL
                    || t == AudioBlockType.PLOSIVE_REDUCTION || t == AudioBlockType.BREATH_REDUCTION);
        }
        AudioBuffer input = new AudioBuffer(noise(9600), new PcmAudioFormat(48000, 2, 16));
        short[] first = new AudioProfileProcessor().process(copy(input), def).getSamples();
        short[] second = new AudioProfileProcessor().process(copy(input), def).getSamples();
        assertArrayEquals(first, second);
    }

    @Test
    public void aProfileWithAllFourBlocksProcessesWithinRange() {
        List<AudioBlockDefinition> blocks = new ArrayList<AudioBlockDefinition>();
        blocks.add(registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "vad"));
        blocks.add(registry.defaultDefinition(AudioBlockType.ADAPTIVE_HUM_REMOVAL, "hum"));
        blocks.add(registry.defaultDefinition(AudioBlockType.PLOSIVE_REDUCTION, "plo"));
        blocks.add(registry.defaultDefinition(AudioBlockType.DE_ESSER, "des"));
        blocks.add(registry.defaultDefinition(AudioBlockType.BREATH_REDUCTION, "bre"));
        AudioProcessingProfile profile = new AudioProcessingProfile("c", "Cleanup", false, blocks);
        AudioBuffer input = new AudioBuffer(tone(1000.0d, 16000, 6000), new PcmAudioFormat(16000, 1, 16));
        short[] out = new AudioProfileProcessor().process(input, profile).getSamples();
        for (short sample : out) {
            assertTrue(sample >= Short.MIN_VALUE && sample <= Short.MAX_VALUE);
        }
    }

    @Test
    public void aDisabledBlockLeavesTheSignalUnchanged() {
        AudioBlockDefinition disabled = registry.defaultDefinition(AudioBlockType.DE_ESSER, "d")
                .withParameter("reductionDb", "20").withEnabled(false);
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "P", false, one(disabled));
        short[] samples = tone(6500.0d, 4000, 9000);
        AudioBuffer input = new AudioBuffer(samples.clone(), new PcmAudioFormat(16000, 1, 16));
        short[] out = new AudioProfileProcessor().process(input, profile).getSamples();
        assertArrayEquals(samples, out);
    }

    private static AudioBuffer copy(AudioBuffer b) {
        return new AudioBuffer(b.getSamples().clone(), b.getFormat());
    }

    private static List<AudioBlockDefinition> one(AudioBlockDefinition block) {
        List<AudioBlockDefinition> list = new ArrayList<AudioBlockDefinition>();
        list.add(block);
        return list;
    }

    private static short[] tone(double freq, int n, int amp) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amp * Math.sin(2.0d * Math.PI * freq * i / 16000.0d));
        }
        return out;
    }

    private static short[] noise(int n) {
        short[] out = new short[n];
        int state = 22222;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            out[i] = (short) (state >> 16);
        }
        return out;
    }
}
