package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.SpeechActivityTrack;
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

/** The VAD block integrates through the registry, never alters the audio, and publishes readable metadata. */
public class VoiceActivityIntegrationTest {

    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void registryCreatesTheVadWithAnalysisCapabilities() {
        assertNotNull(registry.createProcessor(AudioBlockType.VOICE_ACTIVITY_DETECTION));
        AudioBlockCapabilities caps = registry.descriptor(AudioBlockType.VOICE_ACTIVITY_DETECTION).getCapabilities();
        assertFalse("VAD must not modify audio", caps.modifiesAudio());
        assertTrue("VAD produces metadata", caps.producesMetadata());
        assertTrue("VAD requires framing", caps.requiresFraming());
        assertTrue("VAD preserves channel count", caps.preservesChannelCount());
        assertFalse("VAD descriptor exposes parameters",
                registry.descriptor(AudioBlockType.VOICE_ACTIVITY_DETECTION).getParameters().isEmpty());
    }

    @Test
    public void vadPassesMonoAndStereoThroughBitIdentically() {
        assertPassThrough(new AudioBuffer(signal(16000), new PcmAudioFormat(16000, 1, 16)));
        assertPassThrough(new AudioBuffer(stereo(16000), new PcmAudioFormat(48000, 2, 16)));
    }

    @Test
    public void aDisabledVadChangesNothing() {
        AudioBlockDefinition disabled = registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "v")
                .withEnabled(false);
        AudioProcessingProfile profile = new AudioProcessingProfile("d", "Disabled VAD", false, list(disabled));
        short[] samples = signal(8000);
        AudioBuffer input = new AudioBuffer(samples.clone(), new PcmAudioFormat(16000, 1, 16));
        short[] output = new AudioProfileProcessor().process(input, profile).getSamples();
        assertArrayEquals(samples, output);
    }

    @Test
    public void metadataIsPublishedAndReadableByASubsequentStep() {
        AudioBlockProcessor vad = registry.createProcessor(AudioBlockType.VOICE_ACTIVITY_DETECTION);
        AudioBlockDefinition block = registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "v");
        AudioProcessingContext context = new AudioProcessingContext();
        AudioBuffer input = new AudioBuffer(signal(16000), new PcmAudioFormat(16000, 1, 16));

        vad.process(input, block, context);

        SpeechActivityTrack track = context.getSpeechActivity(); // a later block would read this
        assertNotNull("VAD publishes a speech-activity track", track);
        assertEquals("one metadata entry per 20 ms frame", 16000 / 320, track.size());
        for (com.aresstack.audio.dsp.SpeechActivityMetadata frame : track.getFrames()) {
            assertTrue(frame.getSpeechProbability() >= 0.0d && frame.getSpeechProbability() <= 1.0d);
        }
    }

    @Test
    public void metadataDoesNotCarryOverIntoANewRun() {
        AudioBlockProcessor vad = registry.createProcessor(AudioBlockType.VOICE_ACTIVITY_DETECTION);
        AudioBlockDefinition block = registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "v");
        AudioBuffer input = new AudioBuffer(signal(16000), new PcmAudioFormat(16000, 1, 16));

        AudioProcessingContext first = new AudioProcessingContext();
        vad.process(new AudioBuffer(input.getSamples().clone(), input.getFormat()), block, first);
        AudioProcessingContext second = new AudioProcessingContext();
        vad.process(new AudioBuffer(input.getSamples().clone(), input.getFormat()), block, second);

        assertEquals("each run has its own independent track", first.getSpeechActivity().size(),
                second.getSpeechActivity().size());
    }

    @Test
    public void defaultSpeechDoesNotContainTheVad() {
        for (AudioBlockDefinition block : AudioProcessingProfiles.defaultSpeech().getBlocks()) {
            assertFalse(block.getType() == AudioBlockType.VOICE_ACTIVITY_DETECTION);
        }
    }

    private void assertPassThrough(AudioBuffer input) {
        short[] original = input.getSamples().clone();
        AudioBlockProcessor vad = registry.createProcessor(AudioBlockType.VOICE_ACTIVITY_DETECTION);
        AudioBlockDefinition block = registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "v");
        AudioBuffer output = vad.process(input, block, new AudioProcessingContext());
        assertArrayEquals("VAD must not change the samples", original, output.getSamples());
        assertEquals(input.getFormat().getChannels(), output.getFormat().getChannels());
        assertEquals(input.getFormat().getSampleRateHz(), output.getFormat().getSampleRateHz());
    }

    private static short[] signal(int n) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(4000.0d * Math.sin(2.0d * Math.PI * 300.0d * i / 16000.0d));
        }
        return out;
    }

    private static short[] stereo(int frames) {
        short[] out = new short[frames * 2];
        for (int i = 0; i < frames; i++) {
            out[i * 2] = (short) Math.round(4000.0d * Math.sin(2.0d * Math.PI * 300.0d * i / 48000.0d));
            out[i * 2 + 1] = (short) Math.round(2000.0d * Math.sin(2.0d * Math.PI * 600.0d * i / 48000.0d));
        }
        return out;
    }

    private static List<AudioBlockDefinition> list(AudioBlockDefinition block) {
        List<AudioBlockDefinition> blocks = new ArrayList<AudioBlockDefinition>();
        blocks.add(block);
        return blocks;
    }
}
