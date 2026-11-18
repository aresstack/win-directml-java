package com.aresstack.windirectml.inference.whisper.audio;

import java.io.IOException;

/**
 * Delivers sound block by block. A source that ends delivers fewer frames than asked for, and
 * from then on none.
 */
public interface AudioSource extends AutoCloseable {
    /** Allocates what the source needs for this rate and at most this many frames per block. */
    default void prepare(int sampleRate, int maxBlockFrames) {
    }

    /**
     * Fills the first {@link AudioBlock#frames()} frames of the block. Returns how many frames
     * carry sound; the frames after them are set to silence. Less than asked for means the
     * source has ended.
     */
    int read(AudioBlock block) throws IOException;

    @Override
    default void close() throws IOException {
    }
}
