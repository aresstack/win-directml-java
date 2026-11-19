package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.SpeechActivityMetadata;
import com.aresstack.audio.dsp.SpeechActivityTrack;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Speech Gate consumes an upstream VAD track and mutes non-speech to exact digital silence. */
public class SpeechGateTest {

    private static final PcmAudioFormat MONO = new PcmAudioFormat(1000, 1, 16);
    private static final PcmAudioFormat STEREO = new PcmAudioFormat(1000, 2, 16);
    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();

    @Test
    public void registryExposesSpeechGateAsAVadConsumer() {
        assertNotNull(registry.createProcessor(AudioBlockType.SPEECH_GATE));
        AudioBlockCapabilities caps = registry.descriptor(AudioBlockType.SPEECH_GATE).getCapabilities();
        assertTrue(caps.modifiesAudio());
        assertTrue(caps.consumesSpeechMetadata());
        assertTrue(caps.requiresSpeechActivityTrack());
        assertTrue(caps.preservesChannelCount());
        assertFalse(caps.changesDuration());
        assertFalse(caps.changesSampleCount());
    }

    @Test
    public void disabledBlockIsABitIdenticalBypass() {
        short[] samples = filled(400, (short) 1500);
        short[] original = samples.clone();
        AudioBlockDefinition disabled = gate(0, 0, 0, 0).withEnabled(false);
        AudioProcessingProfile profile = new AudioProcessingProfile("p", "p", false, one(disabled));
        short[] out = new AudioProfileProcessor().process(new AudioBuffer(samples, MONO), profile).getSamples();
        assertArrayEquals(original, out);
    }

    @Test
    public void missingTrackPassesThroughAndValidatorErrors() {
        short[] samples = filled(200, (short) 900);
        short[] original = samples.clone();
        registry.createProcessor(AudioBlockType.SPEECH_GATE)
                .process(new AudioBuffer(samples, MONO), gate(0, 0, 0, 0), new AudioProcessingContext());
        assertArrayEquals("defensive bypass without a track", original, samples);

        AudioProcessingProfile gateOnly = new AudioProcessingProfile("g", "g", false, one(gate(0, 0, 0, 0)));
        assertTrue(new AudioProfileValidator().validateResult(gateOnly, MONO).hasErrors());
    }

    @Test
    public void nonSpeechRegionsBecomeExactDigitalSilence() {
        short[] samples = filled(400, (short) 1200);
        run(samples, MONO, gate(0, 0, 0, 0), track(MONO, false, false, false, false));
        assertArrayEquals(new short[400], samples);
    }

    @Test
    public void speechIsKeptAndInternalPausesAreMuted() {
        short[] samples = filled(500, (short) 2000);
        run(samples, MONO, gate(0, 0, 0, 0), track(MONO, false, true, false, true, false));
        assertRange(samples, 0, 100, (short) 0);
        assertRange(samples, 100, 200, (short) 2000);
        assertRange(samples, 200, 300, (short) 0);
        assertRange(samples, 300, 400, (short) 2000);
        assertRange(samples, 400, 500, (short) 0);
    }

    @Test
    public void preRollProtectsWordStarts() {
        short[] samples = filled(300, (short) 2000);
        run(samples, MONO, gate(100, 0, 0, 0), track(MONO, false, true, false));
        assertRange(samples, 0, 100, (short) 2000); // frame before speech kept by pre-roll
        assertRange(samples, 100, 200, (short) 2000);
        assertRange(samples, 200, 300, (short) 0);
    }

    @Test
    public void postRollProtectsWordEnds() {
        short[] samples = filled(300, (short) 2000);
        run(samples, MONO, gate(0, 100, 0, 0), track(MONO, false, true, false));
        assertRange(samples, 0, 100, (short) 0);
        assertRange(samples, 100, 200, (short) 2000);
        assertRange(samples, 200, 300, (short) 2000); // frame after speech kept by post-roll
    }

    @Test
    public void openFadeAvoidsAbruptJumps() {
        short[] samples = filled(300, (short) 2000);
        run(samples, MONO, gate(0, 0, 50, 0), track(MONO, false, true, false));
        boolean intermediate = false;
        for (int i = 100; i < 150; i++) {
            if (samples[i] > 0 && samples[i] < 2000) {
                intermediate = true;
            }
        }
        assertTrue("open fade produces intermediate values, not a 0->full jump", intermediate);
        assertEquals("closed region is exact silence", 0, samples[50]);
    }

    @Test
    public void stereoIsGatedChannelCoupled() {
        short[] samples = filled(400, (short) 2000); // 200 sample-frames x 2 channels
        run(samples, STEREO, gate(0, 0, 0, 0), track(STEREO, true, false));
        assertRange(samples, 0, 200, (short) 2000);   // frame 0 (speech) on both channels
        assertRange(samples, 200, 400, (short) 0);     // frame 1 (silence) on both channels
    }

    @Test
    public void timeBaseChangeBetweenVadAndGateIsRejected() {
        AudioBlockDefinition vad = registry.defaultDefinition(AudioBlockType.VOICE_ACTIVITY_DETECTION, "vad");
        AudioBlockDefinition resampler = registry.defaultDefinition(AudioBlockType.RESAMPLER, "rs");
        AudioBlockDefinition mixer = registry.defaultDefinition(AudioBlockType.CHANNEL_MIXER, "cm");

        AudioProcessingProfile valid = new AudioProcessingProfile("ok", "ok", false,
                Arrays.asList(vad, gate(0, 0, 0, 0)));
        assertFalse("VAD -> Speech Gate is valid", new AudioProfileValidator().validateResult(valid, MONO)
                .hasErrors());

        AudioProcessingProfile invalid = new AudioProcessingProfile("bad", "bad", false,
                Arrays.asList(vad, mixer, resampler, gate(0, 0, 0, 0)));
        assertTrue("a resampler between VAD and Speech Gate is an error",
                new AudioProfileValidator().validateResult(invalid, MONO).hasErrors());
    }

    @Test
    public void defaultProfileDoesNotContainSpeechGate() {
        AudioProcessingProfile def = AudioProcessingProfiles.defaultSpeech();
        for (AudioBlockDefinition block : def.getBlocks()) {
            assertFalse("default profile stays unchanged", block.getType() == AudioBlockType.SPEECH_GATE);
        }
    }

    // ------------------------------------------------------------------ helpers

    private void run(short[] samples, PcmAudioFormat format, AudioBlockDefinition gate,
                     SpeechActivityTrack track) {
        AudioProcessingContext context = new AudioProcessingContext();
        context.setSpeechActivity(track);
        registry.createProcessor(AudioBlockType.SPEECH_GATE)
                .process(new AudioBuffer(samples, format), gate, context);
    }

    private AudioBlockDefinition gate(double pre, double post, double openFade, double closeFade) {
        return registry.defaultDefinition(AudioBlockType.SPEECH_GATE, "gate")
                .withParameter("minSpeechProbability", "0.5")
                .withParameter("preRollMs", Double.toString(pre))
                .withParameter("postRollMs", Double.toString(post))
                .withParameter("attackMs", Double.toString(openFade))
                .withParameter("releaseMs", Double.toString(closeFade));
    }

    private static SpeechActivityTrack track(PcmAudioFormat format, boolean... speech) {
        SpeechActivityTrack track = new SpeechActivityTrack(format.getSampleRateHz(), format.getChannels(), 100);
        for (boolean value : speech) {
            track.add(new SpeechActivityMetadata(value ? 0.9d : 0.1d, value, -60.0d, -20.0d));
        }
        return track;
    }

    private static short[] filled(int count, short value) {
        short[] samples = new short[count];
        Arrays.fill(samples, value);
        return samples;
    }

    private static List<AudioBlockDefinition> one(AudioBlockDefinition block) {
        List<AudioBlockDefinition> list = new ArrayList<AudioBlockDefinition>();
        list.add(block);
        return list;
    }

    private static void assertRange(short[] samples, int start, int end, short expected) {
        for (int i = start; i < end; i++) {
            assertEquals("sample " + i, expected, samples[i]);
        }
    }
}
