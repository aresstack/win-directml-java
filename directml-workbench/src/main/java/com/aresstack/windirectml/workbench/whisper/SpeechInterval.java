package com.aresstack.windirectml.workbench.whisper;

/** Keep a detected speech region in the original 16-kHz recording timeline. */
final class SpeechInterval {
    private final long startSample;
    private final long endSample;

    SpeechInterval(long startSample, long endSample) {
        if (startSample < 0 || endSample <= startSample) {
            throw new IllegalArgumentException("Require a positive speech interval");
        }
        this.startSample = startSample;
        this.endSample = endSample;
    }

    long startSample() {
        return startSample;
    }

    long endSample() {
        return endSample;
    }
}
