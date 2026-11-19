package com.aresstack.audio.dsp;

/**
 * Frequency-dependent adaptive noise suppression in the STFT domain. It tracks the per-bin noise floor with
 * a fast-down/slow-up follower (so stationary and slowly drifting background noise is estimated without a
 * separate detector), then applies a spectral-subtraction gain bounded by the maximum attenuation, phase
 * preserved. Optional speech protection eases the gain and freezes adaptation inside detected speech; a
 * fixed {@link NoiseProfile} replaces the tracker in "use fixed profile" mode. Per-bin gain smoothing
 * (attack/release) plus a light neighbour-bin average limit musical-noise artifacts.
 *
 * <p>One instance is created per channel per run, so the noise estimate never leaks between recordings.</p>
 */
public final class SpectralNoiseSuppressor implements SpectralModifier {

    private static final double OVER_SUBTRACTION = 1.5d;

    private final NoiseSuppressionSettings settings;
    private final NoiseProfile fixedProfile;
    private final SpeechGate speechGate;

    private double[] noiseEst;
    private double[] gain;
    private boolean initialized;

    public SpectralNoiseSuppressor(NoiseSuppressionSettings settings, NoiseProfile fixedProfile,
                                   SpeechGate speechGate) {
        this.settings = settings;
        this.fixedProfile = fixedProfile;
        this.speechGate = speechGate == null ? SpeechGate.NEVER : speechGate;
    }

    public void modify(double[] real, double[] imag, int sampleRateHz, int frameStartSample) {
        int n = real.length;
        int half = n / 2;
        boolean fixed = settings.getMode() == NoiseSuppressionSettings.Mode.USE_FIXED_PROFILE
                && fixedProfile != null && fixedProfile.getFftSize() == n;
        if (!initialized) {
            noiseEst = new double[half + 1];
            gain = new double[half + 1];
            for (int k = 0; k <= half; k++) {
                noiseEst[k] = fixed ? fixedProfile.magnitudeAt(k) : Spectra.magnitude(real, imag, k);
                gain[k] = 1.0d;
            }
            initialized = true;
        }
        boolean speech = settings.isSpeechProtection() && speechGate.isSpeech(frameStartSample + n / 2);
        boolean adapt = !fixed && !settings.isFreezeProfile() && (!speech || settings.isAdaptDuringSpeech());

        double hopMs = 1000.0d * hop(n) / Math.max(1, sampleRateHz);
        double attackCoeff = Spectra.coefficient(settings.getAttackMs(), hopMs);
        double releaseCoeff = Spectra.coefficient(settings.getReleaseMs(), hopMs);
        double fastDown = 0.5d;
        double slowUp = 0.02d + 0.2d * settings.getAdaptationSpeed();
        double floorLin = Math.max(Math.pow(10.0d, -settings.getMaxAttenuationDb() / 20.0d),
                Math.pow(10.0d, settings.getNoiseFloorDb() / 20.0d));
        double protect = speech ? 0.7d : 0.0d;

        double[] target = new double[half + 1];
        for (int k = 1; k <= half; k++) {
            double mag = Spectra.magnitude(real, imag, k);
            if (adapt) {
                double coeff = mag < noiseEst[k] ? fastDown : slowUp;
                noiseEst[k] += coeff * (mag - noiseEst[k]);
            }
            double g = 1.0d - OVER_SUBTRACTION * noiseEst[k] / (mag + Spectra.EPS);
            if (g < floorLin) {
                g = floorLin;
            }
            if (g > 1.0d) {
                g = 1.0d;
            }
            if (protect > 0.0d) {
                g = g + protect * (1.0d - g);
            }
            target[k] = g;
        }
        // Neighbour-bin averaging (artifact protection) followed by attack/release smoothing across frames.
        double ap = settings.getArtifactProtection();
        for (int k = 1; k <= half; k++) {
            double smoothedTarget = target[k];
            if (ap > 0.0d) {
                double lo = target[Math.max(1, k - 1)];
                double hi = target[Math.min(half, k + 1)];
                smoothedTarget = (1.0d - ap) * target[k] + ap * (lo + target[k] + hi) / 3.0d;
            }
            double coeff = smoothedTarget < gain[k] ? attackCoeff : releaseCoeff;
            gain[k] += coeff * (smoothedTarget - gain[k]);
            Spectra.applyGain(real, imag, k, gain[k]);
        }
    }

    private static int hop(int frameSize) {
        return frameSize / 2; // matches the runner's 50% overlap; only used for smoothing time constants
    }
}
