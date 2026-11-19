package com.aresstack.windirectml.inference.piper;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Translate eSpeak IPA code points to Piper's padded phoneme ID sequence.
 * Reject unknown phonemes rather than silently changing the spoken content.
 */
public final class PiperPhonemeIdEncoder {

    private final PiperVoiceConfig voice;

    public PiperPhonemeIdEncoder(PiperVoiceConfig voice) {
        this.voice = Objects.requireNonNull(voice, "voice");
    }

    public long[] encode(String ipa) {
        Objects.requireNonNull(ipa, "ipa");
        if (ipa.isBlank()) {
            throw new IllegalArgumentException("The phoneme sequence must not be blank");
        }
        List<Long> ids = new ArrayList<>();
        append(ids, "^");
        append(ids, "_");
        ipa.codePoints().forEach(point -> {
            String symbol = new String(Character.toChars(point));
            append(ids, symbol);
            append(ids, "_");
        });
        append(ids, "$");
        long[] result = new long[ids.size()];
        for (int index = 0; index < ids.size(); index++) {
            result[index] = ids.get(index);
        }
        return result;
    }

    private void append(List<Long> output, String phoneme) {
        int[] ids = voice.idsFor(phoneme);
        if (ids == null) {
            throw new IllegalArgumentException("Piper voice has no ID for phoneme: " + phoneme);
        }
        for (int id : ids) {
            output.add((long) id);
        }
    }
}
