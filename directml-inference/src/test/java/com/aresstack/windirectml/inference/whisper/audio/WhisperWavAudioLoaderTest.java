package com.aresstack.windirectml.inference.whisper.audio;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class WhisperWavAudioLoaderTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void convertsStereoFortyEightKilohertzWavToMonoSixteenKilohertz() throws Exception {
        Path source = temporaryDirectory.resolve("stereo.wav");
        byte[] pcm = new byte[48000 * 2 * 2];
        for (int i = 0; i < 48000; i++) {
            int offset = i * 4;
            write16(pcm, offset, 16384);
            write16(pcm, offset + 2, -16384);
        }
        writeWav(source, pcm, 48000, 16, 2);

        float[] samples = new WhisperWavAudioLoader().load(source);

        assertEquals(16000, samples.length);
        for (int i = 100; i < samples.length - 100; i += 79) {
            assertEquals(0.0f, samples[i], 1.0e-5f);
        }
    }

    @Test
    void readsTwentyFourBitWavWithoutQuantizingThroughSixteenBit() throws Exception {
        Path source = temporaryDirectory.resolve("high-resolution.wav");
        byte[] pcm = new byte[16000 * 3];
        for (int i = 0; i < 16000; i++) {
            int sample = (i % 2 == 0 ? 1 : -1) * 2097152;
            int offset = i * 3;
            pcm[offset] = (byte) sample;
            pcm[offset + 1] = (byte) (sample >>> 8);
            pcm[offset + 2] = (byte) (sample >>> 16);
        }
        writeWav(source, pcm, 16000, 24, 1);

        float[] samples = new WhisperWavAudioLoader().load(source);

        assertEquals(16000, samples.length);
        assertEquals(0.25f, samples[0], 1.0e-5f);
        assertEquals(-0.25f, samples[1], 1.0e-5f);
    }

    @Test
    void suppressesAliasingWhenDownsampling() {
        float[] highFrequency = sineWave(10000.0, 48000);
        float[] inBand = sineWave(2000.0, 48000);
        float[] filteredHighFrequency = WhisperAudioResampler.resample(highFrequency, 48000, 16000);
        float[] filteredInBand = WhisperAudioResampler.resample(inBand, 48000, 16000);

        assertEquals(16000, filteredHighFrequency.length);
        assertTrue(rms(filteredHighFrequency, 100, 15900) < 0.02);
        assertTrue(rms(filteredInBand, 100, 15900) > 0.65);
    }

    private static float[] sineWave(double frequency, int sampleRate) {
        float[] result = new float[sampleRate];
        for (int i = 0; i < result.length; i++) {
            result[i] = (float) Math.sin(2.0 * Math.PI * frequency * i / sampleRate);
        }
        return result;
    }

    private static double rms(float[] samples, int start, int end) {
        double sum = 0.0;
        for (int i = start; i < end; i++) {
            sum += samples[i] * samples[i];
        }
        return Math.sqrt(sum / (end - start));
    }

    private static void write16(byte[] target, int offset, int sample) {
        target[offset] = (byte) sample;
        target[offset + 1] = (byte) (sample >>> 8);
    }

    private static void writeWav(Path file, byte[] data, int rate, int bits, int channels) throws Exception {
        AudioFormat format = new AudioFormat(rate, bits, channels, true, false);
        try (ByteArrayInputStream input = new ByteArrayInputStream(data);
             AudioInputStream stream = new AudioInputStream(input, format, data.length / format.getFrameSize())) {
            assertTrue(AudioSystem.write(stream, AudioFileFormat.Type.WAVE, file.toFile()) > 0);
        }
    }
}
