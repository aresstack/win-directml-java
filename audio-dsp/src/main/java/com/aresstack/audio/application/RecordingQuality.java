package com.aresstack.audio.application;

/**
 * Business outcome of checking a finished recording before it is sent for transcription.
 *
 * <ul>
 *   <li>{@link #TOO_SHORT} and {@link #NO_SIGNAL} block the upload — there is nothing worth sending.</li>
 *   <li>{@link #CLIPPED} is a warning: the user may transcribe anyway or re-record.</li>
 *   <li>{@link #DROPPED_FRAMES} surfaces that capture could not keep up (never silently ignored).</li>
 *   <li>{@link #VALID} means the recording is good to send.</li>
 * </ul>
 */
public enum RecordingQuality {

    VALID,
    TOO_SHORT,
    NO_SIGNAL,
    CLIPPED,
    DROPPED_FRAMES;

    /** @return true when this outcome must prevent the audio from being uploaded. */
    public boolean blocksUpload() {
        return this == TOO_SHORT || this == NO_SIGNAL;
    }
}
