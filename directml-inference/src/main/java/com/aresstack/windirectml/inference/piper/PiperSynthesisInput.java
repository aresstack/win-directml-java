package com.aresstack.windirectml.inference.piper;

import java.util.Objects;

/**
 * Build the typed Piper VITS input tensors from an eSpeak phoneme sequence.
 * Preserve ONNX input order: input, input_lengths, scales.
 */
public final class PiperSynthesisInput {

    private final long[] phonemeIds;
    private final long[] phonemeLengths;
    private final float[] scales;
    private final int speakerId;

    private PiperSynthesisInput(long[] phonemeIds, float[] scales, int speakerId) {
        this.phonemeIds = phonemeIds.clone();
        this.phonemeLengths = new long[]{phonemeIds.length};
        this.scales = scales.clone();
        this.speakerId = speakerId;
    }

    public static PiperSynthesisInput of(PiperVoiceConfig voice, String ipa, int speakerId) {
        Objects.requireNonNull(voice, "voice");
        if (speakerId < 0 || speakerId >= voice.speakerCount()) {
            throw new IllegalArgumentException("Speaker ID is outside the voice's supported range");
        }
        long[] ids = new PiperPhonemeIdEncoder(voice).encode(ipa);
        return new PiperSynthesisInput(ids,
                new float[]{voice.noiseScale(), voice.lengthScale(), voice.noiseScaleW()},
                speakerId);
    }

    public long[] phonemeIds() {
        return phonemeIds.clone();
    }

    public long[] phonemeLengths() {
        return phonemeLengths.clone();
    }

    public float[] scales() {
        return scales.clone();
    }

    public int speakerId() {
        return speakerId;
    }

    public long[] inputShape() {
        return new long[]{1, phonemeIds.length};
    }

    public long[] scalesShape() {
        return new long[]{3};
    }

    public int phonemeCount() {
        return phonemeIds.length;
    }
}
