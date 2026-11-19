package com.aresstack.audio.dsp;

/**
 * A classic, deterministic, frame-based adaptive voice-activity detector — not a neural model. Per frame it
 * measures RMS energy, peak, zero-crossing rate and crest factor on the strongest channel, tracks an
 * adaptive background-noise floor (with a lower bound so long silence never makes it ever more sensitive),
 * derives a smoothed speech probability from the signal-to-noise ratio and stabilizes the speech-active
 * state with attack, minimum-speech, minimum-silence, hangover and release gating. It never touches the
 * audio; it only produces {@link SpeechActivityMetadata}.
 *
 * <p>The design is deliberately swappable: an alternative detector can produce the same metadata without
 * changing the profile format or the UI.</p>
 */
public final class VoiceActivityDetector {

    private static final double FULL_SCALE = 32768.0d;
    private static final double EPS = 1.0e-9d;
    private static final double NOISE_FLOOR_MIN_LINEAR = 1.0d;   // ~ -90 dBFS: prevents runaway sensitivity
    private static final double LOGISTIC_SCALE_DB = 4.0d;

    /**
     * Analyze one interleaved frame ({@code count} samples starting at {@code offset}) and advance the state.
     */
    public SpeechActivityMetadata analyzeFrame(short[] samples, int offset, int count, int channels,
                                               int sampleRateHz, VoiceActivityDetectorSettings settings,
                                               VoiceActivityDetectorState state) {
        FrameFeatures features = measure(samples, offset, count, Math.max(1, channels));

        if (!state.initialized) {
            state.noiseFloorLinear = Math.max(features.rms, NOISE_FLOOR_MIN_LINEAR);
            state.initialized = true;
        }

        double levelDb = toDb(features.rms);
        double noiseDb = toDb(state.noiseFloorLinear);
        double snrDb = levelDb - noiseDb;

        double rawProbability = rawProbability(snrDb, features, settings);
        double alpha = smoothingAlpha(rawProbability, state.smoothedProbability, settings);
        state.smoothedProbability += alpha * (rawProbability - state.smoothedProbability);
        double probability = clamp01(state.smoothedProbability);

        boolean candidate = probability >= settings.getMinSpeechProbability();
        updateActivity(candidate, settings, state);
        // Adapt the noise floor from the INSTANTANEOUS likelihood, so a loud onset frame is not mistaken
        // for silence before the smoothed probability has caught up (which would raise the floor).
        adaptNoiseFloor(rawProbability, features.rms, settings, state);

        return new SpeechActivityMetadata(probability, state.speechActive,
                toDb(state.noiseFloorLinear), levelDb);
    }

    // ------------------------------------------------------------------ features

    private static FrameFeatures measure(short[] samples, int offset, int count, int channels) {
        FrameFeatures best = new FrameFeatures();
        int end = Math.min(samples.length, offset + count);
        for (int channel = 0; channel < channels; channel++) {
            double sumSquares = 0.0d;
            int peak = 0;
            int samplesInChannel = 0;
            int signChanges = 0;
            int previousSign = 0;
            for (int i = offset + channel; i < end; i += channels) {
                int sample = samples[i];
                sumSquares += (double) sample * sample;
                int abs = Math.abs(sample);
                if (abs > peak) {
                    peak = abs;
                }
                int sign = sample > 0 ? 1 : (sample < 0 ? -1 : 0);
                if (sign != 0) {
                    if (previousSign != 0 && sign != previousSign) {
                        signChanges++;
                    }
                    previousSign = sign;
                }
                samplesInChannel++;
            }
            if (samplesInChannel == 0) {
                continue;
            }
            double rms = Math.sqrt(sumSquares / samplesInChannel);
            if (rms >= best.rms) {
                best.rms = rms;
                best.peak = peak;
                best.zeroCrossingRate = (double) signChanges / samplesInChannel;
                best.crest = peak / (rms + EPS);
            }
        }
        return best;
    }

    // ------------------------------------------------------------------ probability & state

    private static double rawProbability(double snrDb, FrameFeatures features,
                                         VoiceActivityDetectorSettings settings) {
        // Higher sensitivity lowers the SNR required to be considered speech (18 dB → 3 dB).
        double requiredSnrDb = 3.0d + (1.0d - settings.getSensitivity()) * 15.0d;
        double probability = logistic((snrDb - requiredSnrDb) / LOGISTIC_SCALE_DB);
        // Zero-crossing and crest are used as mild plausibility modifiers around the SNR core.
        if (features.zeroCrossingRate > 0.35d) {
            probability *= 0.85d; // very hissy content is less likely to be voiced speech
        }
        if (features.crest > 8.0d) {
            probability *= 0.9d;  // very impulsive frames (clicks) are down-weighted
        }
        return clamp01(probability);
    }

    private static double smoothingAlpha(double rawProbability, double smoothed,
                                         VoiceActivityDetectorSettings settings) {
        double timeMs = rawProbability >= smoothed ? settings.getAttackMs() : settings.getReleaseMs();
        double tau = Math.max(1.0d, timeMs);
        return 1.0d - Math.exp(-settings.getFrameDurationMs() / tau);
    }

    private static void updateActivity(boolean candidate, VoiceActivityDetectorSettings settings,
                                       VoiceActivityDetectorState state) {
        int minSpeechFrames = framesFor(settings.getMinSpeechMs(), settings.getFrameDurationMs());
        int minSilenceFrames = framesFor(settings.getMinSilenceMs(), settings.getFrameDurationMs());
        int hangoverFrames = framesFor(settings.getHangoverMs(), settings.getFrameDurationMs());

        if (!state.speechActive) {
            if (candidate) {
                state.speechRunFrames++;
                if (state.speechRunFrames >= minSpeechFrames) {
                    state.speechActive = true;
                    state.hangoverRemainingFrames = hangoverFrames;
                    state.silenceRunFrames = 0;
                }
            } else {
                state.speechRunFrames = 0;
            }
        } else {
            if (candidate) {
                state.hangoverRemainingFrames = hangoverFrames;
                state.silenceRunFrames = 0;
            } else if (state.hangoverRemainingFrames > 0) {
                state.hangoverRemainingFrames--;
            } else {
                state.silenceRunFrames++;
                if (state.silenceRunFrames >= minSilenceFrames) {
                    state.speechActive = false;
                    state.speechRunFrames = 0;
                }
            }
        }
    }

    private static void adaptNoiseFloor(double rawProbability, double rms,
                                        VoiceActivityDetectorSettings settings,
                                        VoiceActivityDetectorState state) {
        // Freeze while speech is active OR the frame instantaneously looks speech-like (protects the onset
        // before the smoothed decision catches up). A gradually rising noise floor stays below this and is
        // tracked; a sustained loud burst is treated as speech and does not raise the floor.
        boolean looksLikeSpeech = state.speechActive || rawProbability >= 0.5d;
        double speed;
        if (!looksLikeSpeech) {
            speed = settings.getNoiseAdaptationSpeed();          // frame looks like silence: track the noise
        } else if (settings.isAdaptNoiseDuringSpeech()) {
            speed = settings.getNoiseAdaptationSpeed() * 0.1d;   // during speech: adapt only slowly, if at all
        } else {
            return;                                              // never let speech pull the noise floor up
        }
        state.noiseFloorLinear += speed * (rms - state.noiseFloorLinear);
        if (state.noiseFloorLinear < NOISE_FLOOR_MIN_LINEAR) {
            state.noiseFloorLinear = NOISE_FLOOR_MIN_LINEAR;
        }
    }

    // ------------------------------------------------------------------ helpers

    private static int framesFor(double milliseconds, int frameDurationMs) {
        if (milliseconds <= 0.0d) {
            return 0;
        }
        return Math.max(1, (int) Math.round(milliseconds / frameDurationMs));
    }

    private static double toDb(double linear) {
        return 20.0d * Math.log10(Math.max(linear, EPS) / FULL_SCALE);
    }

    private static double logistic(double x) {
        return 1.0d / (1.0d + Math.exp(-x));
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
    }

    /** Per-frame features of the strongest channel. */
    private static final class FrameFeatures {
        double rms;
        int peak;
        double zeroCrossingRate;
        double crest;
    }
}
