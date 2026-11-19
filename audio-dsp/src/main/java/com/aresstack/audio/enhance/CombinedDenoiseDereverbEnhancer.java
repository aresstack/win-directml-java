package com.aresstack.audio.enhance;

import com.aresstack.audio.domain.PcmAudioFormat;
import com.aresstack.audio.dsp.NoiseSuppressionSettings;
import com.aresstack.audio.dsp.SpeechGate;
import com.aresstack.audio.dsp.SpectralBlockRunner;
import com.aresstack.audio.dsp.SpectralModifier;
import com.aresstack.audio.dsp.SpectralNoiseSuppressor;
import com.aresstack.audio.dsp.WpeDereverberation;
import com.aresstack.audio.dsp.WpeDereverberationSettings;

/**
 * Combined pure-Java speech enhancer: dereverberate (offline WPE) and then denoise (adaptive spectral
 * suppression) in one backend, both scaled by the strength. Always available — it reuses the core DSP and
 * needs no native runtime or model. The order (dereverb before denoise) gives the suppressor a cleaner input.
 */
public final class CombinedDenoiseDereverbEnhancer implements SpeechEnhancementBackend {

    private static final int FFT_SIZE = 1024;
    private static final int FFT_HOP = 512;

    public String id() {
        return "COMBINED_DENOISE_DEREVERB";
    }

    public String displayName() {
        return "Combined Denoise + Dereverb (Pure Java)";
    }

    public BackendAvailability availability(PcmAudioFormat format) {
        return BackendAvailability.AVAILABLE;
    }

    public void enhance(short[] samples, int count, PcmAudioFormat format, double strength,
                        final boolean speechProtection, final double artifactProtection) {
        double s = clamp01(strength);
        WpeDereverberationSettings wpe = new WpeDereverberationSettings(
                WpeDereverberationSettings.Mode.OFFLINE, 0.3d + 0.5d * s, 2, 8, 2, 0.5d,
                speechProtection, clamp01(artifactProtection), 0.3d, 64);
        new WpeDereverberation(wpe).process(samples, count, format, SpeechGate.NEVER);

        double maxAttenuationDb = 6.0d + 18.0d * s;
        final NoiseSuppressionSettings ns = new NoiseSuppressionSettings(
                NoiseSuppressionSettings.Mode.AUTOMATIC, maxAttenuationDb, 0.5d, -60.0d,
                speechProtection, 0.5d, false, false, clamp01(artifactProtection), 15.0d, 120.0d);
        SpectralBlockRunner.apply(samples, count, format, FFT_SIZE, FFT_HOP,
                new SpectralBlockRunner.ModifierFactory() {
                    public SpectralModifier create() {
                        return new SpectralNoiseSuppressor(ns, null, null);
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
