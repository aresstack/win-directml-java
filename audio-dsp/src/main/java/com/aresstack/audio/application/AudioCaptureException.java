package com.aresstack.audio.application;

/** Signal a capture problem (device missing, format unsupported) with a user-readable message. */
public class AudioCaptureException extends Exception {

    public AudioCaptureException(String message) {
        super(message);
    }

    public AudioCaptureException(String message, Throwable cause) {
        super(message, cause);
    }
}
