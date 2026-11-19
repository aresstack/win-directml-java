package com.aresstack.windirectml.inference.piper;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/** Write a synthesized Piper waveform as standard little-endian PCM16 WAV. */
public final class PiperWavWriter {

    public void write(PiperAudio audio, Path destination) throws IOException {
        Objects.requireNonNull(audio, "audio");
        Objects.requireNonNull(destination, "destination");
        Path parent = destination.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        float[] values = audio.samples();
        byte[] pcm = new byte[Math.multiplyExact(values.length, Short.BYTES)];
        for (int i = 0; i < values.length; i++) {
            float bounded = Math.min(1f, Math.max(-1f, values[i]));
            short sample = (short) Math.round(bounded * 32767f);
            pcm[i * 2] = (byte) sample;
            pcm[i * 2 + 1] = (byte) (sample >>> 8);
        }

        AudioFormat format = new AudioFormat(audio.sampleRate(), 16, 1, true, false);
        try (AudioInputStream stream = new AudioInputStream(
                new ByteArrayInputStream(pcm), format, values.length)) {
            AudioSystem.write(stream, AudioFileFormat.Type.WAVE, destination.toFile());
        }
    }
}
