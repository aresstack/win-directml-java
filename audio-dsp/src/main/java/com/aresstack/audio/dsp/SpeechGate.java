package com.aresstack.audio.dsp;

/** Answers whether a given mono sample index falls in detected speech (bridges the STFT to the VAD track). */
public interface SpeechGate {

    SpeechGate NEVER = new SpeechGate() {
        public boolean isSpeech(int monoSampleIndex) {
            return false;
        }
    };

    boolean isSpeech(int monoSampleIndex);
}
