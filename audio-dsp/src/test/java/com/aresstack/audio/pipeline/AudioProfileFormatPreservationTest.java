package com.aresstack.audio.pipeline;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.profile.AudioBlockDefinition;
import com.aresstack.audio.profile.AudioBlockType;
import com.aresstack.audio.profile.AudioProcessingProfile;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Pin the format-neutrality guarantee: the pipeline preserves the input sample rate and channel count
 * unless an explicit structural block (RESAMPLER / channel mixer) changes it. Guards against a return of
 * the old implicit 16 kHz / mono normalization.
 */
public class AudioProfileFormatPreservationTest {

    private final AudioBlockRegistry registry = AudioBlockRegistry.getInstance();
    private final AudioProfileProcessor processor = new AudioProfileProcessor();

    @Test
    public void emptyProfilePreserves44100Stereo() {
        AudioBuffer in = new AudioBuffer(stereoTone(44100), new PcmAudioFormat(44100, 2, 16));
        AudioBuffer out = processor.process(in, profile("off"));
        assertEquals(44100, out.getFormat().getSampleRateHz());
        assertEquals(2, out.getFormat().getChannels());
        assertEquals(in.getSamples().length, out.getSamples().length);
    }

    @Test
    public void gainPreserves48000Stereo() {
        AudioBuffer in = new AudioBuffer(stereoTone(48000), new PcmAudioFormat(48000, 2, 16));
        AudioBuffer out = processor.process(in, profile("gain", block(AudioBlockType.GAIN, "g")));
        assertEquals(48000, out.getFormat().getSampleRateHz());
        assertEquals(2, out.getFormat().getChannels());
    }

    @Test
    public void equalizerPreservesRateAndChannels() {
        AudioBuffer in = new AudioBuffer(stereoTone(44100), new PcmAudioFormat(44100, 2, 16));
        AudioBuffer out = processor.process(in, profile("eq", block(AudioBlockType.LOW_SHELF, "eq")));
        assertEquals(44100, out.getFormat().getSampleRateHz());
        assertEquals(2, out.getFormat().getChannels());
    }

    @Test
    public void resamplerChangesOnlyTheRateToItsConfiguredValue() {
        // Resampler requires mono input; use 8000 Hz target to prove the rate is not hardcoded to 16000.
        AudioBuffer in = new AudioBuffer(monoTone(44100), new PcmAudioFormat(44100, 1, 16));
        AudioBlockDefinition resampler = block(AudioBlockType.RESAMPLER, "r").withParameter("targetRateHz", "8000");
        AudioBuffer out = processor.process(in, profile("resample", resampler));
        assertEquals(8000, out.getFormat().getSampleRateHz());
        assertEquals(1, out.getFormat().getChannels());
    }

    @Test
    public void channelMixerChangesOnlyTheChannelCount() {
        AudioBuffer in = new AudioBuffer(stereoTone(48000), new PcmAudioFormat(48000, 2, 16));
        AudioBuffer out = processor.process(in, profile("mono", block(AudioBlockType.CHANNEL_MIXER, "m")));
        assertEquals(48000, out.getFormat().getSampleRateHz());
        assertEquals(1, out.getFormat().getChannels());
    }

    private AudioBlockDefinition block(AudioBlockType type, String id) {
        return registry.defaultDefinition(type, id);
    }

    private static AudioProcessingProfile profile(String id, AudioBlockDefinition... blocks) {
        List<AudioBlockDefinition> list = new ArrayList<AudioBlockDefinition>();
        for (AudioBlockDefinition block : blocks) {
            list.add(block);
        }
        return new AudioProcessingProfile(id, id, false, list);
    }

    private static short[] monoTone(int frames) {
        short[] samples = new short[frames];
        for (int i = 0; i < frames; i++) {
            samples[i] = (short) Math.round(6000.0d * Math.sin(2.0d * Math.PI * 200.0d * i / 16000.0d));
        }
        return samples;
    }

    private static short[] stereoTone(int frames) {
        short[] samples = new short[frames * 2];
        for (int frame = 0; frame < frames; frame++) {
            short value = (short) Math.round(6000.0d * Math.sin(2.0d * Math.PI * 200.0d * frame / 16000.0d));
            samples[frame * 2] = value;
            samples[frame * 2 + 1] = (short) (value / 2);
        }
        return samples;
    }
}
