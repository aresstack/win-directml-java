package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Reduce audible breath sounds. Breaths are the audible, non-speech stretches between words: this block
 * uses the upstream speech-activity track to attenuate non-speech frames whose level rises above the
 * estimated noise floor, by up to the maximum attenuation and scaled by sensitivity, while protecting
 * detected speech. Without a track it passes the audio through unchanged (the validator warns), never
 * duplicating a detector. Time-domain, broadband attenuation; a fresh instance per run isolates state.
 */
public final class BreathReductionProcessor {

    private static final double FULL_SCALE = 32768.0d;
    private static final double EPS = 1.0e-9d;
    private static final double ACTIVATION_RANGE_DB = 15.0d;

    private final BreathReductionSettings settings;

    private double gainDb;
    private boolean initialized;

    public BreathReductionProcessor(BreathReductionSettings settings) {
        if (settings == null) {
            throw new IllegalArgumentException("Breath-reduction settings must not be null.");
        }
        this.settings = settings;
    }

    /** Process interleaved PCM16 in place. {@code track} may be null (then the block is a pass-through). */
    public void process(short[] samples, int count, PcmAudioFormat format, SpeechActivityTrack track) {
        if (track == null) {
            return; // no speech metadata: never guess breath from level alone
        }
        int channels = Math.max(1, format.getChannels());
        int rate = format.getSampleRateHz();
        double attackCoeff = coefficient(settings.getAttackMs(), rate);
        double releaseCoeff = coefficient(settings.getReleaseMs(), rate);
        // Higher sensitivity → the level needs to rise less above the noise floor to count as breath.
        double marginDb = 12.0d - settings.getSensitivity() * 10.0d;

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
            if (!initialized) {
                gainDb = 0.0d;
                initialized = true;
            }
            SpeechActivityMetadata frame = track.frameForInterleavedIndex(base);
            double targetGainDb = 0.0d;
            // With speech protection on, only non-speech frames may be attenuated; without it, breath is
            // judged purely by level above the noise floor.
            boolean candidate = frame != null && (!settings.isSpeechProtection() || !frame.isSpeechActive());
            if (candidate) {
                double levelDb = 20.0d * Math.log10(Math.max(peak, EPS) / FULL_SCALE);
                double above = levelDb - (frame.getEstimatedNoiseLevelDb() + marginDb);
                double activation = clamp01(above / ACTIVATION_RANGE_DB);
                targetGainDb = -settings.getMaxAttenuationDb() * activation;
            }
            double coeff = targetGainDb < gainDb ? attackCoeff : releaseCoeff;
            gainDb += coeff * (targetGainDb - gainDb);
            double linear = Math.pow(10.0d, gainDb / 20.0d);
            for (int c = 0; c < channels; c++) {
                samples[base + c] = clamp(samples[base + c] * linear);
            }
        }
    }

    private static double coefficient(double milliseconds, int rate) {
        double samples = Math.max(1.0d, milliseconds * rate / 1000.0d);
        return 1.0d - Math.exp(-1.0d / samples);
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || value < 0.0d) {
            return 0.0d;
        }
        return value > 1.0d ? 1.0d : value;
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
