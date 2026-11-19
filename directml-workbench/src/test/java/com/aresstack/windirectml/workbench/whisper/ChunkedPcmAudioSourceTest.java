package com.aresstack.windirectml.workbench.whisper;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChunkedPcmAudioSourceTest {

    @Test
    void spillsThePreviousChunkOnlyWhenTheNextTwoMinuteChunkIsComplete() throws Exception {
        int sourceRate = 16000;
        int fullChunkBytes = sourceRate * 120 * 2;

        try (ChunkedPcmAudioSource recording = new ChunkedPcmAudioSource(sourceRate)) {
            recording.append(constantPcm(fullChunkBytes, (short) 8192), 0, fullChunkBytes);
            assertEquals(0, recording.spilledChunkCount());
            assertEquals(fullChunkBytes, recording.residentAudioBytes());

            byte[] secondChunk = constantPcm(fullChunkBytes, (short) -8192);
            recording.append(secondChunk, 0, fullChunkBytes / 2);
            assertEquals(0, recording.spilledChunkCount());
            assertEquals(fullChunkBytes + fullChunkBytes / 2, recording.residentAudioBytes());

            recording.append(secondChunk, fullChunkBytes / 2, fullChunkBytes / 2);
            assertEquals(1, recording.spilledChunkCount());
            assertEquals(fullChunkBytes, recording.residentAudioBytes());

            byte[] finalChunk = constantPcm(16000 * 8 * 2, (short) 16384);
            recording.append(finalChunk, 0, finalChunk.length);
            recording.finish();

            assertEquals(248 * 16000L, recording.sampleCount());
            assertEquals(1, recording.spilledChunkCount());
            assertEquals(fullChunkBytes + finalChunk.length, recording.residentAudioBytes());

            float[] crossingFirstBoundary = recording.readSamples(120 * 16000L - 4, 8);
            for (int i = 0; i < 4; i++) {
                assertEquals(0.25f, crossingFirstBoundary[i], 0.0001f);
                assertEquals(-0.25f, crossingFirstBoundary[i + 4], 0.0001f);
            }

            float[] crossingSecondBoundary = recording.readSamples(240 * 16000L - 4, 8);
            for (int i = 0; i < 4; i++) {
                assertEquals(-0.25f, crossingSecondBoundary[i], 0.0001f);
                assertEquals(0.5f, crossingSecondBoundary[i + 4], 0.0001f);
            }
        }
    }

    @Test
    void readsUnalignedSixteenKilohertzWindowsWithoutChangingTheirPosition() throws Exception {
        try (ChunkedPcmAudioSource recording = new ChunkedPcmAudioSource(16000)) {
            byte[] input = constantPcm(32000 * 2, (short) 4096);
            recording.append(input, 0, input.length);
            recording.finish();

            float[] window = recording.readSamples(177, 234);
            assertEquals(234, window.length);
            for (float sample : window) {
                assertEquals(0.125f, sample, 0.0001f);
            }
            assertThrows(IllegalArgumentException.class,
                    () -> recording.readSamples(recording.sampleCount() - 10, 11));
        }
    }

    @Test
    void resamplesFortyFourKilohertzAudioWithoutReadingTheEntireRecording() throws Exception {
        try (ChunkedPcmAudioSource recording = new ChunkedPcmAudioSource(44100)) {
            byte[] input = constantPcm(44100 * 3 * 2, (short) 8192);
            recording.append(input, 0, input.length);
            recording.finish();

            assertEquals(48000, recording.sampleCount());
            float[] window = recording.readSamples(16000 + 123, 500);
            assertEquals(500, window.length);
            for (float sample : window) {
                assertTrue(Math.abs(sample - 0.25f) < 0.001f);
            }
        }
    }

    private static byte[] constantPcm(int byteCount, short sample) {
        byte[] pcm = new byte[byteCount];
        for (int i = 0; i < byteCount; i += 2) {
            pcm[i] = (byte) sample;
            pcm[i + 1] = (byte) (sample >>> 8);
        }
        return pcm;
    }
}
