package com.aresstack.windirectml.inference.piper;

/**
 * Convert language-specific text to eSpeak-compatible phonemes.
 * Keep the phonemization implementation separate from model inference.
 */
public interface PiperPhonemizer {

    String phonemize(String text, String espeakVoice) throws Exception;
}
