package com.aresstack.audio.enhance;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * A pluggable speech-enhancement backend. Pure-Java backends are always available; native/model backends
 * (RNNoise, DeepFilterNet, …) report their availability so the block can stay editable and pass audio
 * through unchanged when the runtime or model is missing, instead of failing. Backends must not depend on
 * Swing/AWT and must be safe to instantiate without their native runtime present.
 */
public interface SpeechEnhancementBackend {

    /** Stable identifier stored in the profile (never a display string). */
    String id();

    String displayName();

    /** @return whether this backend can actually run for the given format right now. */
    BackendAvailability availability(PcmAudioFormat format);

    /**
     * Enhance the interleaved PCM16 in place. Only called when {@link #availability(PcmAudioFormat)} is
     * runnable. {@code strength} in [0,1] scales the effect; the protection flags limit over-processing.
     */
    void enhance(short[] samples, int count, PcmAudioFormat format, double strength, boolean speechProtection,
                 double artifactProtection);
}
