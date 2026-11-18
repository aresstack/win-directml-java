package com.aresstack.windirectml.workbench.whisper;

import com.aresstack.windirectml.inference.whisper.audio.WhisperAudioResampler;

/** Convert captured signed little-endian PCM16 to 16 kHz normalized float audio. */
final class MicrophonePcmConverter {
    private MicrophonePcmConverter() {
    }

    static float[] convert(byte[] captured, int sourceRate) {
        if (captured == null || (captured.length & 1) != 0 || sourceRate <= 0) {
            throw new IllegalArgumentException("Require complete PCM16 frames and a positive sample rate");
        }
        float[] normalized = new float[captured.length / 2];
        for (int frame = 0; frame < normalized.length; frame++) {
            int offset = frame * 2;
            short signed = (short) ((captured[offset] & 0xFF) | (captured[offset + 1] << 8));
            normalized[frame] = signed / 32768.0f;
        }
        return WhisperAudioResampler.resample(normalized, sourceRate, 16000);
    }
}
