package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Speech-aware automatic level control: it stabilizes the perceived speech level while a talker moves
 * nearer to or further from the microphone. Gain is driven by the smoothed speech level (not the raw signal
 * level), so during detected silence the gain is not pulled up to lift background noise. It respects a
 * maximum boost/cut, a maximum gain change per second, a hold time, an optional noise-aware boost limit and
 * clipping protection. Channel-linked: one gain is applied to every channel of a frame.
 *
 * <p>Stateful; a fresh {@link SpeechLevelerState} per run makes the result reproducible for identical input
 * and settings. Without an upstream speech-activity track it falls back to a level threshold (the validator
 * warns).</p>
 */
public final class SpeechLevelerProcessor {

    private static final double FULL_SCALE = 32768.0d;
    private static final double CEILING = 32767.0d;
    private static final double EPS = 1.0e-9d;
    private static final double LEVEL_BASED_SPEECH_FLOOR_DB = -45.0d;

    private final SpeechLevelerSettings settings;

    public SpeechLevelerProcessor(SpeechLevelerSettings settings) {
        if (settings == null) {
            throw new IllegalArgumentException("Speech leveler settings must not be null.");
        }
        this.settings = settings;
    }

    public void process(short[] samples, int count, PcmAudioFormat format, SpeechLevelerState state,
                        SpeechActivityTrack track) {
        int channels = Math.max(1, format.getChannels());
        int rate = format.getSampleRateHz();
        double envCoeff = coefficient(15.0d, rate);
        double attackCoeff = coefficient(settings.getAttackMs(), rate);
        double releaseCoeff = coefficient(settings.getReleaseMs(), rate);
        double backgroundCoeff = coefficient(500.0d, rate);
        double maxDeltaPerSampleDb = settings.getMaxGainChangePerSecond() / rate;
        int holdSamples = (int) Math.round(settings.getHoldMs() * rate / 1000.0d);
        double targetDb = settings.getTargetSpeechLevelDb();

        int frames = count / channels;
        for (int f = 0; f < frames; f++) {
            int base = f * channels;
            double peak = 0.0d;
            double meanSquare = 0.0d;
            for (int c = 0; c < channels; c++) {
                double v = samples[base + c];
                meanSquare += v * v;
                double abs = Math.abs(v);
                if (abs > peak) {
                    peak = abs;
                }
            }
            meanSquare /= channels;
            state.speechEnvelope += envCoeff * (meanSquare - state.speechEnvelope);
            double rms = Math.sqrt(Math.max(0.0d, state.speechEnvelope));
            double levelDb = 20.0d * Math.log10(Math.max(rms, EPS) / FULL_SCALE);

            boolean speech = isSpeech(track, base, levelDb);
            double desiredGainDb;
            if (speech) {
                state.speechLevelDb = levelDb;
                desiredGainDb = targetDb - levelDb;
                double maxGain = settings.getMaxGainDb();
                if (settings.isNoiseProtection() && !Double.isNaN(state.backgroundLevelDb)) {
                    // Do not boost so much that the tracked background floor is lifted above the target.
                    maxGain = Math.min(maxGain, targetDb - state.backgroundLevelDb);
                    if (maxGain < 0.0d) {
                        maxGain = 0.0d;
                    }
                }
                desiredGainDb = clamp(desiredGainDb, -settings.getMaxAttenuationDb(), maxGain);
                state.holdSamplesRemaining = holdSamples;
            } else {
                state.backgroundLevelDb = Double.isNaN(state.backgroundLevelDb) ? levelDb
                        : state.backgroundLevelDb + backgroundCoeff * (levelDb - state.backgroundLevelDb);
                if (state.holdSamplesRemaining > 0) {
                    state.holdSamplesRemaining--;
                    desiredGainDb = state.currentGainDb; // hold the last speech gain briefly
                } else {
                    // Return toward the silence limit; never raise gain during silence.
                    desiredGainDb = Math.min(state.currentGainDb, settings.getSilenceGainLimitDb());
                }
            }

            double coeff = desiredGainDb < state.currentGainDb ? attackCoeff : releaseCoeff;
            double step = coeff * (desiredGainDb - state.currentGainDb);
            if (step > maxDeltaPerSampleDb) {
                step = maxDeltaPerSampleDb;
            } else if (step < -maxDeltaPerSampleDb) {
                step = -maxDeltaPerSampleDb;
            }
            state.currentGainDb += step;

            if (settings.isClippingProtection() && peak > 0.0d) {
                double maxGainNoClipDb = 20.0d * Math.log10(CEILING / peak);
                if (state.currentGainDb > maxGainNoClipDb) {
                    state.currentGainDb = maxGainNoClipDb; // immediate, before it can overshoot
                }
            }

            double linear = Math.pow(10.0d, state.currentGainDb / 20.0d);
            for (int c = 0; c < channels; c++) {
                samples[base + c] = clamp(samples[base + c] * linear);
            }
        }
    }

    private boolean isSpeech(SpeechActivityTrack track, int interleavedIndex, double levelDb) {
        if (track == null) {
            return levelDb > LEVEL_BASED_SPEECH_FLOOR_DB; // level-based fallback (validator warns)
        }
        SpeechActivityMetadata frame = track.frameForInterleavedIndex(interleavedIndex);
        if (frame == null) {
            return false;
        }
        return frame.getSpeechProbability() >= settings.getMinSpeechProbability() || frame.isSpeechActive();
    }

    private static double coefficient(double milliseconds, int rate) {
        double samples = Math.max(1.0d, milliseconds * rate / 1000.0d);
        return 1.0d - Math.exp(-1.0d / samples);
    }

    private static double clamp(double value, double min, double max) {
        if (Double.isNaN(value)) {
            return 0.0d;
        }
        return value < min ? min : (value > max ? max : value);
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
