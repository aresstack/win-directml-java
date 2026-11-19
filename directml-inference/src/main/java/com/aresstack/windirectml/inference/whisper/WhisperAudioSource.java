package com.aresstack.windirectml.inference.whisper;

import java.io.IOException;

/**
 * Supply 16-kHz mono PCM in independently readable windows.
 * Keep source storage independent of the Whisper inference pipeline.
 */
public interface WhisperAudioSource {
    long sampleCount();

    /**
     * Read up to the requested number of samples starting at an absolute position.
     * Return the complete requested interval when it is inside the recording.
     */
    float[] readSamples(long startSample, int sampleCount) throws IOException;
}
