package com.aresstack.windirectml.inference.piper;

import java.io.IOException;
import java.util.Objects;

/**
 * Synthesize a mono Piper waveform with the complete native Java VITS reference graph.
 * Keep this CPU implementation separate from the future DirectML GPU backend.
 */
public final class PiperReferenceSynthesizer {

    private final PiperModelImport voice;
    private final PiperNativeGraphPrefixRunner executor;

    public PiperReferenceSynthesizer(PiperModelImport voice) {
        this(voice, 42L);
    }

    public PiperReferenceSynthesizer(PiperModelImport voice, long randomSeed) {
        this.voice = Objects.requireNonNull(voice, "voice");
        this.executor = new PiperNativeGraphPrefixRunner(randomSeed);
    }

    public PiperAudio synthesizePhonemes(String ipa) throws IOException {
        return synthesizePhonemes(ipa, 0);
    }

    public PiperAudio synthesizePhonemes(String ipa, int speakerId) throws IOException {
        PiperSynthesisInput input = PiperSynthesisInput.of(voice.voice(), ipa, speakerId);
        PiperNativeGraphPrefixRunner.PrefixResult result =
                executor.execute(voice, input, voice.nodeCount());
        return extractMonoWaveform(result.output());
    }

    public PiperAudio synthesizeText(String text, PiperPhonemizer phonemizer) throws Exception {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(phonemizer, "phonemizer");
        if (text.isBlank()) {
            throw new IllegalArgumentException("Piper text must not be blank");
        }
        String phonemes = phonemizer.phonemize(text, voice.voice().espeakVoice());
        return synthesizePhonemes(phonemes);
    }

    private PiperAudio extractMonoWaveform(PiperTensor output) {
        int[] dimensions = output.shape();
        if (output.dataType() != PiperTensor.FLOAT32 || dimensions.length < 2) {
            throw new IllegalStateException("Piper VITS output must be a floating point waveform");
        }
        for (int i = 0; i < dimensions.length - 1; i++) {
            if (dimensions[i] != 1) {
                throw new IllegalStateException("Piper reference synthesis supports one mono voice");
            }
        }
        return new PiperAudio(voice.voice().sampleRate(), output.floatData());
    }
}
