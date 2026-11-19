package com.aresstack.audio.infrastructure;

import com.aresstack.audio.application.AudioSink;
import com.aresstack.audio.domain.PcmAudioFormat;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * Write PCM frames into a standard RIFF/WAVE file. Write a placeholder header first, stream the
 * data chunk, and patch the RIFF and data sizes on {@link #close()} so any standard audio tool
 * can play the result.
 */
public final class WavFileAudioSink implements AudioSink {

    private static final int HEADER_SIZE_BYTES = 44;

    private final File file;
    private RandomAccessFile output;
    private PcmAudioFormat format;
    private long dataBytesWritten;
    private byte[] encodeBuffer = new byte[0];

    public WavFileAudioSink(File file) {
        if (file == null) {
            throw new IllegalArgumentException("Output file must not be null.");
        }
        this.file = file;
    }

    @Override
    public void open(PcmAudioFormat format) throws IOException {
        this.format = format;
        this.dataBytesWritten = 0;
        File parent = file.getParentFile();
        if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
            throw new IOException("Could not create output directory: " + parent.getAbsolutePath());
        }
        output = new RandomAccessFile(file, "rw");
        output.setLength(0);
        writeHeader(0);
    }

    @Override
    public void write(short[] samples, int count) throws IOException {
        ensureOpen();
        int byteCount = count * 2;
        if (encodeBuffer.length < byteCount) {
            encodeBuffer = new byte[byteCount];
        }
        Pcm16LittleEndianCodec.encode(samples, count, encodeBuffer);
        output.write(encodeBuffer, 0, byteCount);
        dataBytesWritten += byteCount;
    }

    @Override
    public void close() throws IOException {
        if (output == null) {
            return;
        }
        // Patch the actual sizes into the header now that the data length is known.
        output.seek(0);
        writeHeader(dataBytesWritten);
        output.close();
        output = null;
    }

    private void ensureOpen() throws IOException {
        if (output == null) {
            throw new IOException("WAV sink is not open: " + file.getAbsolutePath());
        }
    }

    /** Write the 44-byte canonical PCM WAV header for the given data size. */
    private void writeHeader(long dataBytes) throws IOException {
        int channels = format.getChannels();
        int sampleRate = format.getSampleRateHz();
        int bitsPerSample = format.getBitsPerSample();
        int blockAlign = channels * bitsPerSample / 8;
        int byteRate = sampleRate * blockAlign;

        output.writeBytes("RIFF");
        writeIntLe(output, (int) (36 + dataBytes));
        output.writeBytes("WAVE");
        output.writeBytes("fmt ");
        writeIntLe(output, 16);                 // fmt chunk size
        writeShortLe(output, 1);                // audio format: PCM
        writeShortLe(output, channels);
        writeIntLe(output, sampleRate);
        writeIntLe(output, byteRate);
        writeShortLe(output, blockAlign);
        writeShortLe(output, bitsPerSample);
        output.writeBytes("data");
        writeIntLe(output, (int) dataBytes);
    }

    private static void writeIntLe(RandomAccessFile out, int value) throws IOException {
        out.write(value & 0xFF);
        out.write((value >> 8) & 0xFF);
        out.write((value >> 16) & 0xFF);
        out.write((value >> 24) & 0xFF);
    }

    private static void writeShortLe(RandomAccessFile out, int value) throws IOException {
        out.write(value & 0xFF);
        out.write((value >> 8) & 0xFF);
    }
}
