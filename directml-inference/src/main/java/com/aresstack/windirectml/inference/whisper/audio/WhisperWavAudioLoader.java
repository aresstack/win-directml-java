package com.aresstack.windirectml.inference.whisper.audio;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;

/**
 * Load a WAV source into the normalized 16-kHz mono float PCM required by Whisper.
 * Adapt 16-, 24-, 32-bit integer and floating WAV sources without a native decoder.
 */
public final class WhisperWavAudioLoader {
    private static final int BLOCK_FRAMES = 8192;

    public float[] load(Path wavFile) throws IOException {
        Objects.requireNonNull(wavFile, "wavFile");
        try (WavFileSource source = new WavFileSource(wavFile)) {
            int sourceRate = Math.round(source.sampleRate());
            AudioBlock block = new AudioBlock(BLOCK_FRAMES);
            source.prepare(sourceRate, BLOCK_FRAMES);
            SampleCollector collector = new SampleCollector();
            int framesRead;
            do {
                block.setFrames(BLOCK_FRAMES);
                framesRead = source.read(block);
                collector.appendMono(block, framesRead);
            } while (framesRead == BLOCK_FRAMES);
            return WhisperAudioResampler.resample(
                    collector.toArray(), sourceRate, WhisperLogMelSpectrogram.SAMPLE_RATE);
        }
    }

    private static final class SampleCollector {
        private float[] samples = new float[BLOCK_FRAMES];
        private int length;

        void appendMono(AudioBlock block, int count) {
            growIfNecessary(count);
            for (int i = 0; i < count; i++) {
                samples[length++] = (block.left[i] + block.right[i]) * 0.5f;
            }
        }

        float[] toArray() {
            return Arrays.copyOf(samples, length);
        }

        private void growIfNecessary(int added) {
            long required = (long) length + added;
            if (required > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Audio recording exceeds Java array size");
            }
            if (required <= samples.length) {
                return;
            }
            int capacity = samples.length;
            while (capacity < required) {
                capacity = capacity <= Integer.MAX_VALUE / 2 ? capacity * 2 : Integer.MAX_VALUE;
            }
            samples = Arrays.copyOf(samples, capacity);
        }
    }
}
