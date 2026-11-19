package com.aresstack.audio.infrastructure;

import com.aresstack.audio.domain.PcmAudioFormat;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class WavFileAudioSinkTest {

    private static final int HEADER_SIZE = 44;

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void writesPlayablePcmWavFile() throws IOException {
        PcmAudioFormat format = PcmAudioFormat.speechDefault();
        File file = new File(temporaryFolder.getRoot(), "test.wav");
        WavFileAudioSink sink = new WavFileAudioSink(file);

        // Write 100 ms of a 440 Hz tone in two frames.
        int frameSize = format.samplesForMillis(50);
        short[] samples = new short[frameSize];
        sink.open(format);
        for (int frame = 0; frame < 2; frame++) {
            for (int i = 0; i < frameSize; i++) {
                samples[i] = (short) (10000 * Math.sin(2 * Math.PI * 440 * i / format.getSampleRateHz()));
            }
            sink.write(samples, frameSize);
        }
        sink.close();

        long expectedDataBytes = 2L * frameSize * 2 /* frames */;
        assertTrue("File must exist", file.isFile());
        assertEquals("File size must be header + data", HEADER_SIZE + expectedDataBytes, file.length());

        byte[] header = readBytes(file, HEADER_SIZE);
        assertEquals("RIFF", new String(header, 0, 4, "US-ASCII"));
        assertEquals("WAVE", new String(header, 8, 4, "US-ASCII"));
        assertEquals("data", new String(header, 36, 4, "US-ASCII"));
        assertEquals("PCM format tag", 1, readShortLe(header, 20));
        assertEquals("Channel count", format.getChannels(), readShortLe(header, 22));
        assertEquals("Sample rate", format.getSampleRateHz(), readIntLe(header, 24));
        assertEquals("Bits per sample", format.getBitsPerSample(), readShortLe(header, 34));
        assertEquals("Data chunk size", expectedDataBytes, readIntLe(header, 40));
    }

    private static byte[] readBytes(File file, int count) throws IOException {
        byte[] bytes = new byte[count];
        FileInputStream input = new FileInputStream(file);
        try {
            int offset = 0;
            while (offset < count) {
                int read = input.read(bytes, offset, count - offset);
                if (read < 0) {
                    throw new IOException("File shorter than " + count + " bytes.");
                }
                offset += read;
            }
        } finally {
            input.close();
        }
        return bytes;
    }

    private static int readShortLe(byte[] bytes, int offset) {
        return (bytes[offset] & 0xFF) | ((bytes[offset + 1] & 0xFF) << 8);
    }

    private static long readIntLe(byte[] bytes, int offset) {
        return (bytes[offset] & 0xFFL) | ((bytes[offset + 1] & 0xFFL) << 8)
                | ((bytes[offset + 2] & 0xFFL) << 16) | ((bytes[offset + 3] & 0xFFL) << 24);
    }
}
