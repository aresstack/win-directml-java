package com.aresstack.audio.enhance;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * RNNoise backend adapter. RNNoise is a native library and is not bundled with the pure-Java core, so this
 * adapter reports its availability from a probe (a system property pointing at an installed native runtime)
 * and reports {@code NOT_INSTALLED} otherwise. It never fails to instantiate and never runs unless actually
 * available. RNNoise is designed for 48 kHz mono; other rates are reported as an invalid sample rate.
 */
public final class RnnoiseSpeechEnhancer implements SpeechEnhancementBackend {

    public static final String PROPERTY = "askai.audio.rnnoise.runtime";

    public String id() {
        return "RNNOISE";
    }

    public String displayName() {
        return "RNNoise";
    }

    public BackendAvailability availability(PcmAudioFormat format) {
        String runtime = System.getProperty(PROPERTY);
        if (runtime == null || runtime.trim().length() == 0) {
            return BackendAvailability.NOT_INSTALLED;
        }
        if (format != null && format.getSampleRateHz() != 48000) {
            return BackendAvailability.INVALID_SAMPLE_RATE;
        }
        return BackendAvailability.NOT_INSTALLED; // no JNI binding bundled — see problems.md
    }

    public void enhance(short[] samples, int count, PcmAudioFormat format, double strength,
                        boolean speechProtection, double artifactProtection) {
        throw new UnsupportedOperationException("RNNoise native runtime is not available.");
    }
}
