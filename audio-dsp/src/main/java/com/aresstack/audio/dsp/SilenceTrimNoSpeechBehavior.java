package com.aresstack.audio.dsp;

/** What the silence trimmer does when no speech is detected in the whole recording. */
public enum SilenceTrimNoSpeechBehavior {
    /** Keep the original signal untouched (default: never surprise the user with an empty file). */
    KEEP_ORIGINAL,
    /** Fail in a controlled way so the caller can surface the problem. */
    FAIL
}
