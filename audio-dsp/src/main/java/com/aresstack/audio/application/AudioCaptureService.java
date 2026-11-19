package com.aresstack.audio.application;

import com.aresstack.audio.dsp.AudioProcessingPipeline;

import java.io.IOException;

/**
 * Record for a fixed duration. Thin wrapper around {@link SpeechRecordingSession}, which holds
 * the actual plumbing (frame segmentation, bounded queue, writer thread).
 */
public final class AudioCaptureService {

    /**
     * Capture from the source for the given duration, process every frame through the pipeline
     * and write the result to the sink. Block until finished.
     *
     * @return the number of frames dropped because the sink could not keep up
     */
    public long capture(AudioSource source, AudioProcessingPipeline pipeline, AudioSink sink,
                        int frameDurationMillis, long durationMillis)
            throws AudioCaptureException, IOException {
        SpeechRecordingSession session =
                new SpeechRecordingSession(source, sink, pipeline, frameDurationMillis);
        session.start();
        try {
            Thread.sleep(durationMillis);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
        return session.stop();
    }
}
