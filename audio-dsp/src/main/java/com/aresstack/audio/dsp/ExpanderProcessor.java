package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * A soft downward expander: below the threshold the dynamics are progressively widened (quiet parts are
 * attenuated), limited by a maximum attenuation; above the threshold the signal is passed through (apart
 * from the attack/release smoothing of the gain). This is a continuous expander, distinct from the hard
 * Noise Gate — it never replaces it.
 *
 * <p>The detector is channel-linked (the strongest channel per frame drives one gain applied to all
 * channels, so the stereo image is not shifted). When speech protection is on and an upstream
 * {@link SpeechActivityTrack} is available, detected speech keeps the expander open; without a track the
 * expander still runs purely level-based and never duplicates a speech detector.</p>
 */
public final class ExpanderProcessor {

    private static final double FULL_SCALE = 32768.0d;
    private static final double EPS = 1.0e-9d;

    private final ExpanderSettings settings;

    public ExpanderProcessor(ExpanderSettings settings) {
        if (settings == null) {
            throw new IllegalArgumentException("Expander settings must not be null.");
        }
        this.settings = settings;
    }

    /** Process interleaved PCM16 in place. {@code track} may be null when no speech protection applies. */
    public void process(short[] samples, int count, PcmAudioFormat format, ExpanderState state,
                        SpeechActivityTrack track) {
        int channels = Math.max(1, format.getChannels());
        int rate = format.getSampleRateHz();
        double envCoeff = coefficient(settings.getDetectorWindowMs(), rate);
        double attackCoeff = coefficient(settings.getAttackMs(), rate);
        double releaseCoeff = coefficient(settings.getReleaseMs(), rate);
        int holdSamples = (int) Math.round(settings.getHoldMs() * rate / 1000.0d);
        boolean useSpeech = settings.isSpeechProtection() && track != null;

        int frames = count / channels;
        for (int f = 0; f < frames; f++) {
            int base = f * channels;
            double peak = 0.0d;
            for (int c = 0; c < channels; c++) {
                int abs = Math.abs(samples[base + c]);
                if (abs > peak) {
                    peak = abs;
                }
            }
            if (!state.initialized) {
                state.envelopeLinear = peak;
                state.initialized = true;
            }
            state.envelopeLinear += envCoeff * (peak - state.envelopeLinear);

            double levelDb = 20.0d * Math.log10(Math.max(state.envelopeLinear, EPS) / FULL_SCALE);
            double targetGainDb = staticGainDb(levelDb);
            if (useSpeech && isSpeechFrame(track, base)) {
                targetGainDb = 0.0d; // protect detected speech: keep fully open
            }

            state.gainDb = smoothGain(state, targetGainDb, attackCoeff, releaseCoeff, holdSamples);

            double linearGain = Math.pow(10.0d, state.gainDb / 20.0d);
            for (int c = 0; c < channels; c++) {
                samples[base + c] = clamp(samples[base + c] * linearGain);
            }
        }
    }

    /** The static downward-expansion gain (dB, <= 0) with a soft knee and a maximum attenuation. */
    private double staticGainDb(double levelDb) {
        double threshold = settings.getThresholdDb();
        double knee = settings.getKneeDb();
        double ratio = settings.getRatio();
        double over = levelDb - threshold;
        double gainDb;
        if (over >= knee / 2.0d) {
            gainDb = 0.0d;
        } else if (knee > 0.0d && over > -knee / 2.0d) {
            double d = threshold + knee / 2.0d - levelDb; // 0..knee across the knee
            gainDb = -(ratio - 1.0d) * d * d / (2.0d * knee);
        } else {
            gainDb = (ratio - 1.0d) * (levelDb - threshold); // fully below: negative
        }
        double maxAtt = -settings.getMaxAttenuationDb();
        return gainDb < maxAtt ? maxAtt : gainDb;
    }

    private double smoothGain(ExpanderState state, double target, double attackCoeff, double releaseCoeff,
                              int holdSamples) {
        double gainDb = state.gainDb;
        if (target > gainDb) {
            // Opening (gain rising toward 0): react with the attack time and arm the hold.
            gainDb += attackCoeff * (target - gainDb);
            state.holdRemainingSamples = holdSamples;
        } else if (state.holdRemainingSamples > 0) {
            state.holdRemainingSamples--; // one frame elapsed; hold the current gain (no closing yet)
        } else {
            // Closing (more attenuation): react with the release time.
            gainDb += releaseCoeff * (target - gainDb);
        }
        return gainDb;
    }

    private static boolean isSpeechFrame(SpeechActivityTrack track, int interleavedIndex) {
        SpeechActivityMetadata frame = track.frameForInterleavedIndex(interleavedIndex);
        return frame != null && frame.isSpeechActive();
    }

    private static double coefficient(double milliseconds, int rate) {
        double samples = Math.max(1.0d, milliseconds * rate / 1000.0d);
        return 1.0d - Math.exp(-1.0d / samples);
    }

    private static short clamp(double value) {
        if (Double.isNaN(value)) {
            return 0;
        }
        if (value > Short.MAX_VALUE) {
            return Short.MAX_VALUE;
        }
        if (value < Short.MIN_VALUE) {
            return Short.MIN_VALUE;
        }
        return (short) Math.round(value);
    }
}
