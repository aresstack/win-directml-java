package com.aresstack.audio.application;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Deliver raw PCM sample blocks from some capture device. Implementations do the capturing only —
 * no DSP, no file or network output. Blocks may arrive in arbitrary sizes; segmentation into
 * fixed frames is the caller's concern.
 */
public interface AudioSource {

    /** Receive raw sample blocks as they are captured; the array is reused between calls. */
    interface SampleListener {
        void onSamples(short[] samples, int count);
    }

    /** @return the PCM format this source captures in. */
    PcmAudioFormat getFormat();

    /**
     * Start capturing and push sample blocks to the listener from a capture thread until
     * {@link #stop()} is called.
     *
     * @throws AudioCaptureException when the device is missing or does not support the format
     */
    void start(SampleListener listener) throws AudioCaptureException;

    /** Stop capturing and release the device. Safe to call more than once. */
    void stop();
}
