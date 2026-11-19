package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioSystem;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class PiperAudioTest {

    @TempDir
    Path directory;

    @Test
    void copiesTheWaveformAndReportsSampleRateAndDuration() {
        float[] samples = new float[]{0f, 0.5f, -0.25f};
        PiperAudio output = new PiperAudio(22_050, samples);
        samples[0] = 1f;
        output.samples()[1] = -1f;

        assertArrayEquals(new float[]{0f, 0.5f, -0.25f}, output.samples(), 0f);
        assertEquals(22_050, output.sampleRate());
        assertEquals(3, output.sampleCount());
        assertEquals(3d / 22_050, output.durationSeconds(), 0);
    }

    @Test
    void writesReadableSixteenBitMonoWav() throws Exception {
        Path file = directory.resolve("synthesized.wav");
        PiperAudio audio = new PiperAudio(22_050,
                new float[]{-1f, -0.5f, 0f, 0.5f, 1f});

        new PiperWavWriter().write(audio, file);

        AudioFileFormat format = AudioSystem.getAudioFileFormat(file.toFile());
        assertEquals(AudioFileFormat.Type.WAVE, format.getType());
        assertEquals(22_050f, format.getFormat().getSampleRate());
        assertEquals(16, format.getFormat().getSampleSizeInBits());
        assertEquals(1, format.getFormat().getChannels());
        assertEquals(5, format.getFrameLength());
        assertTrue(java.nio.file.Files.size(file) > 44);
    }

    @Test
    void rejectsNaNAndEmptyWaveforms() {
        assertThrows(IllegalArgumentException.class, () ->
                new PiperAudio(22_050, new float[0]));
        assertThrows(IllegalArgumentException.class, () ->
                new PiperAudio(22_050, new float[]{Float.NaN}));
        assertThrows(IllegalArgumentException.class, () ->
                new PiperAudio(22_050, new float[]{1.3f}));
    }
}
