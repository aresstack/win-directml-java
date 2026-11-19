package com.aresstack.audio.enhance;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * DeepFilterNet backend adapter. DeepFilterNet is a neural model that needs a native/ONNX runtime plus a
 * model file, neither of which is bundled with the pure-Java core. The adapter probes runtime and model
 * system properties and reports {@code NOT_INSTALLED} / {@code MISSING_MODEL} / {@code INVALID_SAMPLE_RATE}
 * so the block stays editable; it never runs unless actually available.
 */
public final class DeepFilterNetSpeechEnhancer implements SpeechEnhancementBackend {

    public static final String RUNTIME_PROPERTY = "askai.audio.deepfilternet.runtime";
    public static final String MODEL_PROPERTY = "askai.audio.deepfilternet.model";

    public String id() {
        return "DEEPFILTERNET";
    }

    public String displayName() {
        return "DeepFilterNet";
    }

    public BackendAvailability availability(PcmAudioFormat format) {
        String runtime = System.getProperty(RUNTIME_PROPERTY);
        if (runtime == null || runtime.trim().length() == 0) {
            return BackendAvailability.NOT_INSTALLED;
        }
        String model = System.getProperty(MODEL_PROPERTY);
        if (model == null || model.trim().length() == 0) {
            return BackendAvailability.MISSING_MODEL;
        }
        if (format != null && format.getSampleRateHz() != 48000) {
            return BackendAvailability.INVALID_SAMPLE_RATE;
        }
        return BackendAvailability.NOT_INSTALLED; // no runtime binding bundled — see problems.md
    }

    public void enhance(short[] samples, int count, PcmAudioFormat format, double strength,
                        boolean speechProtection, double artifactProtection) {
        throw new UnsupportedOperationException("DeepFilterNet runtime/model is not available.");
    }
}
