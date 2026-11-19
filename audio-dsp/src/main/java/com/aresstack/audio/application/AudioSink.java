package com.aresstack.audio.application;

import com.aresstack.audio.domain.PcmAudioFormat;

import java.io.IOException;

/**
 * Receive finished PCM frames after processing. Implementations write files, keep samples in
 * memory for tests, or stream to a remote speech-to-text system later — capture and DSP code
 * never knows which.
 */
public interface AudioSink {

    /** Prepare the sink for samples of the given format. Call once before the first write. */
    void open(PcmAudioFormat format) throws IOException;

    /** Append the given samples; only the first {@code count} entries are valid. */
    void write(short[] samples, int count) throws IOException;

    /** Finish and release the sink (e.g. patch WAV header sizes). Safe to call more than once. */
    void close() throws IOException;
}
