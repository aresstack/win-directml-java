package com.aresstack.windirectml.inference.whisper.audio;

import java.io.IOException;
import java.nio.file.Path;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * Reads a WAV file block by block, at its own sample rate: whole numbers of 16, 24 or 32 bits
 * and 32-bit floats as they are, without rounding them first; other formats through the
 * conversion of Java Sound to 16 bits. A mono file is heard on both sides. Only one block of
 * the file is in memory at a time.
 */
public final class WavFileSource implements AudioSource {
    private final AudioInputStream input;
    private final AudioInputStream opened;
    private final AudioFormat format;
    private final boolean floating;
    private final int channels;
    private final int bytes;
    private final int frameSize;
    private byte[] buffer = new byte[0];
    private int pending;
    private boolean ended;

    public WavFileSource(Path file) throws IOException {
        AudioInputStream stream;
        try {
            stream = AudioSystem.getAudioInputStream(file.toFile());
        } catch (UnsupportedAudioFileException unsupported) {
            throw new IOException("Not a readable WAV file: " + file, unsupported);
        }
        opened = stream;
        try {
            AudioFormat found = stream.getFormat();
            int bits = found.getSampleSizeInBits();
            boolean whole = AudioFormat.Encoding.PCM_SIGNED.equals(found.getEncoding())
                    && (bits == 16 || bits == 24 || bits == 32);
            boolean isFloat = AudioFormat.Encoding.PCM_FLOAT.equals(found.getEncoding()) && bits == 32;
            if (found.getChannels() < 1 || found.getChannels() > 2 || !(whole || isFloat)) {
                int count = Math.min(2, Math.max(1, found.getChannels()));
                AudioFormat target = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED, found.getSampleRate(), 16,
                        count, count * 2, found.getSampleRate(), false);
                if (!AudioSystem.isConversionSupported(target, found)) {
                    throw new IOException("Unsupported WAV sample format for " + file + ": " + found);
                }
                input = AudioSystem.getAudioInputStream(target, stream);
                format = target;
                floating = false;
            } else {
                input = stream;
                format = found;
                floating = isFloat;
            }
        } catch (IOException | RuntimeException failed) {
            stream.close();
            throw failed;
        }
        channels = format.getChannels();
        bytes = format.getSampleSizeInBits() / 8;
        frameSize = bytes * channels;
    }

    /** The sample rate of the file. */
    public float sampleRate() {
        return format.getSampleRate();
    }

    /** Counts the frames of a file by reading it through, for files whose head does not say. */
    public static long countFrames(Path file) throws IOException {
        try (WavFileSource source = new WavFileSource(file)) {
            AudioBlock block = new AudioBlock(16384);
            source.prepare((int) source.sampleRate(), block.capacity());
            long frames = 0;
            int got;
            do {
                block.setFrames(block.capacity());
                got = source.read(block);
                frames += got;
            } while (got == block.capacity());
            return frames;
        }
    }

    @Override
    public void prepare(int sampleRate, int maxBlockFrames) {
        buffer = new byte[Math.max(16384, maxBlockFrames * frameSize + frameSize)];
        pending = 0;
    }

    @Override
    public int read(AudioBlock block) throws IOException {
        int wanted = block.frames();
        int count = 0;
        while (count < wanted && !ended) {
            int whole = pending / frameSize;
            if (whole == 0) {
                int got = input.read(buffer, pending, buffer.length - pending);
                if (got < 0) {
                    ended = true; // a part of a frame at the end is dropped, as before
                    break;
                }
                pending += got;
                continue;
            }
            int take = Math.min(whole, wanted - count);
            boolean bigEndian = format.isBigEndian();
            for (int i = 0; i < take; i++) {
                int offset = i * frameSize;
                float left = sample(buffer, offset, bytes, bigEndian, floating);
                block.left[count + i] = left;
                block.right[count + i] = channels == 2
                        ? sample(buffer, offset + bytes, bytes, bigEndian, floating) : left;
            }
            int used = take * frameSize;
            System.arraycopy(buffer, used, buffer, 0, pending - used);
            pending -= used;
            count += take;
        }
        block.clear(count, wanted - count);
        return count;
    }

    private static float sample(byte[] data, int offset, int bytes, boolean bigEndian, boolean floating) {
        int value = 0;
        for (int i = 0; i < bytes; i++) {
            int part = data[offset + (bigEndian ? bytes - 1 - i : i)] & 255;
            value |= part << (8 * i);
        }
        if (floating) {
            return Float.intBitsToFloat(value);
        }
        int shift = 32 - 8 * bytes;
        return (value << shift >> shift) / (float) (1L << (8 * bytes - 1));
    }

    @Override
    public void close() throws IOException {
        try {
            input.close();
        } finally {
            opened.close();
        }
    }
}
