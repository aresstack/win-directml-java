package com.aresstack.audio.infrastructure;

import com.aresstack.audio.application.AudioSink;
import com.aresstack.audio.domain.PcmAudioFormat;

import java.io.IOException;

/**
 * Collect every written sample in memory. Meant for unit tests and for later hand-off of a whole
 * recording to an upload; not for unbounded recordings.
 */
public final class MemoryAudioSink implements AudioSink {

    private short[] buffer = new short[16384];
    private int size;
    private PcmAudioFormat format;
    private boolean open;

    @Override
    public void open(PcmAudioFormat format) {
        this.format = format;
        this.size = 0;
        this.open = true;
    }

    @Override
    public void write(short[] samples, int count) throws IOException {
        if (!open) {
            throw new IOException("Memory sink is not open.");
        }
        ensureCapacity(size + count);
        System.arraycopy(samples, 0, buffer, size, count);
        size += count;
    }

    @Override
    public void close() {
        open = false;
    }

    public PcmAudioFormat getFormat() {
        return format;
    }

    public int getSampleCount() {
        return size;
    }

    /** Copy the recorded samples into a right-sized array. */
    public short[] toArray() {
        short[] copy = new short[size];
        System.arraycopy(buffer, 0, copy, 0, size);
        return copy;
    }

    private void ensureCapacity(int needed) {
        if (needed <= buffer.length) {
            return;
        }
        int newLength = buffer.length;
        while (newLength < needed) {
            newLength *= 2;
        }
        short[] grown = new short[newLength];
        System.arraycopy(buffer, 0, grown, 0, size);
        buffer = grown;
    }
}
