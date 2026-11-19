package com.aresstack.audio.enhance;

import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.NoiseSuppressionSettings;
import com.aresstack.audio.dsp.SpectralBlockRunner;
import com.aresstack.audio.dsp.SpectralModifier;
import com.aresstack.audio.dsp.SpectralNoiseSuppressor;

/**
 * Built-in, always-available speech enhancer: adaptive spectral noise suppression scaled by the strength.
 * It needs no native runtime or model, so the Speech Enhancer block is functional out of the box and the
 * pure-Java core never depends on an optional backend.
 */
public final class PureJavaSpeechEnhancer implements SpeechEnhancementBackend {

    private static final int FFT_SIZE = 1024;
    private static final int FFT_HOP = 512;

    public String id() {
        return "PURE_JAVA_DSP";
    }

    public String displayName() {
        return "Pure Java Adaptive DSP";
    }

    public BackendAvailability availability(PcmAudioFormat format) {
        return BackendAvailability.AVAILABLE;
    }

    public void enhance(short[] samples, int count, PcmAudioFormat format, final double strength,
                        final boolean speechProtection, final double artifactProtection) {
        double maxAttenuationDb = 6.0d + 18.0d * clamp01(strength);
        final NoiseSuppressionSettings settings = new NoiseSuppressionSettings(
                NoiseSuppressionSettings.Mode.AUTOMATIC, maxAttenuationDb, 0.5d, -60.0d,
                speechProtection, 0.5d, false, false, clamp01(artifactProtection), 15.0d, 120.0d);
        SpectralBlockRunner.apply(samples, count, format, FFT_SIZE, FFT_HOP,
                new SpectralBlockRunner.ModifierFactory() {
                    public SpectralModifier create() {
                        return new SpectralNoiseSuppressor(settings, null, null);
                    }
                });
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
    }
}
