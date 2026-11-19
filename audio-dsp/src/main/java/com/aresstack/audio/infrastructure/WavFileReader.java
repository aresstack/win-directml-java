package com.aresstack.audio.infrastructure;

import com.aresstack.audio.domain.PcmAudioFormat;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * Read a PCM RIFF/WAVE file into 16-bit samples plus its {@link PcmAudioFormat}. The counterpart of
 * {@link WavFileAudioSink}. Parses the {@code fmt } chunk (channels, sample rate, bits) and the
 * {@code data} chunk, skipping any other chunks (LIST/fact/…). Recordings are small, so the file is
 * read fully into memory. Only 16-bit PCM is supported (matches this module's contract).
 */
public final class WavFileReader {

    /** The samples of a WAV file together with the format they were stored in. */
    public static final class WavData {
        private final PcmAudioFormat format;
        private final short[] samples;

        public WavData(PcmAudioFormat format, short[] samples) {
            this.format = format;
            this.samples = samples;
        }

        public PcmAudioFormat getFormat() {
            return format;
        }

        /** @return the interleaved 16-bit samples (one per channel per frame). */
        public short[] getSamples() {
            return samples;
        }
    }

    private WavFileReader() {
    }

    public static WavData read(File file) throws IOException {
        byte[] bytes = readAllBytes(file);
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes));
        try {
            if (!"RIFF".equals(readTag(in))) {
                throw new IOException("Not a RIFF file: " + file);
            }
            skipFully(in, 4); // RIFF chunk size
            if (!"WAVE".equals(readTag(in))) {
                throw new IOException("Not a WAVE file: " + file);
            }

            int channels = 0;
            int sampleRate = 0;
            int bitsPerSample = 0;
            byte[] dataBytes = null;

            while (true) {
                String chunkId = readTagOrNull(in);
                if (chunkId == null) {
                    break; // end of file
                }
                int chunkSize = readIntLe(in);
                if ("fmt ".equals(chunkId)) {
                    skipFully(in, 2); // audio format (assume PCM)
                    channels = readShortLe(in);
                    sampleRate = readIntLe(in);
                    skipFully(in, 4); // byte rate
                    skipFully(in, 2); // block align
                    bitsPerSample = readShortLe(in);
                    skipFully(in, chunkSize - 16); // any extra fmt bytes
                } else if ("data".equals(chunkId)) {
                    dataBytes = new byte[chunkSize];
                    in.readFully(dataBytes);
                } else {
                    skipFully(in, chunkSize);
                }
                if ((chunkSize & 1) == 1) {
                    skipFully(in, 1); // chunks are word-aligned
                }
            }

            if (channels <= 0 || sampleRate <= 0 || bitsPerSample != 16) {
                throw new IOException("Unsupported or missing PCM fmt chunk in " + file
                        + " (channels=" + channels + ", rate=" + sampleRate + ", bits=" + bitsPerSample + ")");
            }
            if (dataBytes == null) {
                dataBytes = new byte[0];
            }
            short[] samples = new short[dataBytes.length / 2];
            Pcm16LittleEndianCodec.decode(dataBytes, dataBytes.length, samples);
            return new WavData(new PcmAudioFormat(sampleRate, channels, bitsPerSample), samples);
        } finally {
            in.close();
        }
    }

    private static byte[] readAllBytes(File file) throws IOException {
        RandomAccessFile raf = new RandomAccessFile(file, "r");
        try {
            long length = raf.length();
            if (length > Integer.MAX_VALUE) {
                throw new IOException("WAV file too large: " + file);
            }
            byte[] bytes = new byte[(int) length];
            raf.readFully(bytes);
            return bytes;
        } finally {
            raf.close();
        }
    }

    private static String readTag(DataInputStream in) throws IOException {
        String tag = readTagOrNull(in);
        if (tag == null) {
            throw new EOFException("Unexpected end of WAV file.");
        }
        return tag;
    }

    private static String readTagOrNull(DataInputStream in) throws IOException {
        byte[] tag = new byte[4];
        int read = 0;
        while (read < 4) {
            int r = in.read(tag, read, 4 - read);
            if (r < 0) {
                return null;
            }
            read += r;
        }
        return new String(tag, "US-ASCII");
    }

    private static int readIntLe(DataInputStream in) throws IOException {
        int b0 = in.readUnsignedByte();
        int b1 = in.readUnsignedByte();
        int b2 = in.readUnsignedByte();
        int b3 = in.readUnsignedByte();
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }

    private static int readShortLe(DataInputStream in) throws IOException {
        int b0 = in.readUnsignedByte();
        int b1 = in.readUnsignedByte();
        return (short) (b0 | (b1 << 8));
    }

    private static void skipFully(DataInputStream in, int count) throws IOException {
        int remaining = count;
        while (remaining > 0) {
            long skipped = in.skip(remaining);
            if (skipped <= 0) {
                if (in.read() < 0) {
                    throw new EOFException("Unexpected end of WAV file while skipping.");
                }
                remaining--;
            } else {
                remaining -= (int) skipped;
            }
        }
    }
}
