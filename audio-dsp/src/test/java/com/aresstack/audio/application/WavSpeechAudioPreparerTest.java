package com.aresstack.audio.application;

import com.aresstack.audio.domain.AudioBuffer;
import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.infrastructure.WavFileReader;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** The STT preparer writes the audio to WAV preserving its rate/channels — no forced down-mix or resample. */
public class WavSpeechAudioPreparerTest {

    @Rule
    public final TemporaryFolder folder = new TemporaryFolder();

    private final WavSpeechAudioPreparer preparer = new WavSpeechAudioPreparer();

    @Test
    public void keeps48kStereoUnchanged() throws Exception {
        short[] samples = stereoTone(48000);
        File out = preparer.prepare(new AudioBuffer(samples, new PcmAudioFormat(48000, 2, 16)),
                folder.newFile("s48.wav"));

        WavFileReader.WavData data = WavFileReader.read(out);
        assertEquals(48000, data.getFormat().getSampleRateHz());
        assertEquals(2, data.getFormat().getChannels());
        assertEquals(16, data.getFormat().getBitsPerSample());
        assertArrayEquals("samples written verbatim", samples, data.getSamples());
    }

    @Test
    public void keeps16kMonoUnchanged() throws Exception {
        short[] samples = monoTone(16000);
        WavFileReader.WavData data = WavFileReader.read(
                preparer.prepare(new AudioBuffer(samples, new PcmAudioFormat(16000, 1, 16)),
                        folder.newFile("m16.wav")));
        assertEquals(16000, data.getFormat().getSampleRateHz());
        assertEquals(1, data.getFormat().getChannels());
        assertArrayEquals(samples, data.getSamples());
    }

    @Test
    public void keeps44kMonoUnchanged() throws Exception {
        WavFileReader.WavData data = WavFileReader.read(
                preparer.prepare(new AudioBuffer(monoTone(44100), new PcmAudioFormat(44100, 1, 16)),
                        folder.newFile("m44.wav")));
        assertEquals(44100, data.getFormat().getSampleRateHz());
        assertEquals(1, data.getFormat().getChannels());
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
