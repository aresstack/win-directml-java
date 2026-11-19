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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Expander and Silence Trimmer integrate through the registry and honor the VAD-driven contract. */
public class ExpanderSilenceTrimmerIntegrationTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void registryCreatesBothBlocksWithParameters() {
        for (AudioBlockType type : new AudioBlockType[]{AudioBlockType.EXPANDER, AudioBlockType.SILENCE_TRIMMER}) {
            assertNotNull(registry.createProcessor(type));
            assertFalse(registry.descriptor(type).getParameters().isEmpty());
        }
        AudioBlockCapabilities trimmer = registry.descriptor(AudioBlockType.SILENCE_TRIMMER).getCapabilities();
        assertTrue(trimmer.changesDuration());
        assertTrue(trimmer.requiresSpeechActivityTrack());
        assertFalse(trimmer.supportsStreaming());
    }

    @Test
    public void defaultSpeechContainsNeitherBlockAndStaysDeterministic() {
        AudioProcessingProfile def = AudioProcessingProfiles.defaultSpeech();
        for (AudioBlockDefinition block : def.getBlocks()) {
            assertFalse(block.getType() == AudioBlockType.EXPANDER
                    || block.getType() == AudioBlockType.SILENCE_TRIMMER);
        }
        AudioBuffer input = new AudioBuffer(tone(300.0d, 9600, 8000), new PcmAudioFormat(48000, 2, 16));
        short[] first = new AudioProfileProcessor().process(copy(input), def).getSamples();
        short[] second = new AudioProfileProcessor().process(copy(input), def).getSamples();
        assertArrayEquals(first, second);
    }

    @Test
    public void vadThenSilenceTrimmerRemovesLeadingAndTrailingSilence() {
        short[] signal = concat(new short[RATE / 2], tone(300.0d, RATE / 2, 9000), new short[RATE / 2]);
        AudioProcessingProfile profile = profile(vad(), trimmer(true, true, "KEEP_ORIGINAL"));
        AudioBuffer out = new AudioProfileProcessor().process(new AudioBuffer(signal, MONO), profile);
        assertTrue("trimmed shorter than the original", out.getSamples().length < signal.length);
        assertTrue("keeps roughly the speech region", out.getSamples().length >= RATE / 2);
        assertEquals(1, out.getFormat().getChannels());
    }

    @Test
    public void silenceTrimmerWithoutUpstreamVadPassesAudioThrough() {
        short[] signal = concat(new short[RATE / 2], tone(300.0d, RATE / 2, 9000));
        AudioProcessingProfile profile = profile(trimmer(true, true, "KEEP_ORIGINAL"));
        AudioBuffer out = new AudioProfileProcessor().process(new AudioBuffer(signal.clone(), MONO), profile);
        assertArrayEquals("no track → no trim, no energy fallback", signal, out.getSamples());
    }

    @Test
    public void noSpeechWithFailBehaviorRaisesAControlledError() {
        AudioProcessingProfile profile = profile(vad(), trimmer(true, true, "FAIL"));
        try {
            new AudioProfileProcessor().process(new AudioBuffer(new short[RATE], MONO), profile);
            fail("expected a controlled failure when no speech is present");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().toLowerCase().contains("no speech"));
        }
    }

    @Test
    public void disabledTrimmerLeavesTheSignalUnchanged() {
        short[] signal = concat(new short[RATE / 2], tone(300.0d, RATE / 2, 9000), new short[RATE / 2]);
        AudioBlockDefinition disabled = trimmer(true, true, "KEEP_ORIGINAL").withEnabled(false);
        AudioProcessingProfile profile = profile(vad(), disabled);
        AudioBuffer out = new AudioProfileProcessor().process(new AudioBuffer(signal.clone(), MONO), profile);
        assertArrayEquals(signal, out.getSamples());
    }

    // ------------------------------------------------------------------ helpers

    private AudioBlockDefinition vad() {
        return registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "vad");
    }

    private AudioBlockDefinition trimmer(boolean lead, boolean trail, String noSpeech) {
        return registry.defaultDefinition(AudioBlockType.SILENCE_TRIMMER, "trim")
                .withParameter("trimLeading", String.valueOf(lead))
                .withParameter("trimTrailing", String.valueOf(trail))
                .withParameter("noSpeechBehavior", noSpeech)
                .withParameter("minRetainedMs", "0");
    }

    private static AudioProcessingProfile profile(AudioBlockDefinition... blocks) {
        List<AudioBlockDefinition> list = new ArrayList<AudioBlockDefinition>();
        for (AudioBlockDefinition block : blocks) {
            list.add(block);
        }
        return new AudioProcessingProfile("p", "P", false, list);
    }

    private static AudioBuffer copy(AudioBuffer buffer) {
        return new AudioBuffer(buffer.getSamples().clone(), buffer.getFormat());
    }

    private static short[] tone(double freq, int n, int amplitude) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amplitude * Math.sin(2.0d * Math.PI * freq * i / RATE));
        }
        return out;
    }

    private static short[] concat(short[]... parts) {
        int total = 0;
        for (short[] part : parts) {
            total += part.length;
        }
        short[] out = new short[total];
        int offset = 0;
        for (short[] part : parts) {
            System.arraycopy(part, 0, out, offset, part.length);
            offset += part.length;
        }
        return out;
    }
}
