package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * A channel-strip equalizer: a low shelf, a parametric mid band and a high shelf for tone shaping, followed
 * by a master gain and an optional loudness stage. The loudness stage is the sound-technically correct way to
 * make a signal louder and fuller: it folds the make-up gain into a soft (tanh) saturation curve that lifts
 * low/mid levels and limits peaks toward a ceiling — increasing perceived loudness without hard clipping,
 * rather than boosting peaks (which would only add clipping). The tone bands reuse the shared biquad EQ
 * processors; a fresh instance is created per run so filter state never leaks between recordings.
 */
public final class MasterEqualizerProcessor implements Pcm16Processor {

    private static final double FULL_SCALE = 32768.0d;

    private final double lowShelfHz;
    private final double lowShelfGainDb;
    private final double midHz;
    private final double midGainDb;
    private final double midQ;
    private final double highShelfHz;
    private final double highShelfGainDb;
    private final double masterGainDb;
    private final boolean loudness;
    private final double loudnessDriveDb;
    private final double ceilingDb;

    public MasterEqualizerProcessor(double lowShelfHz, double lowShelfGainDb, double midHz, double midGainDb,
                                    double midQ, double highShelfHz, double highShelfGainDb, double masterGainDb,
                                    boolean loudness, double loudnessDriveDb, double ceilingDb) {
        this.lowShelfHz = lowShelfHz;
        this.lowShelfGainDb = lowShelfGainDb;
        this.midHz = midHz;
        this.midGainDb = midGainDb;
        this.midQ = midQ;
        this.highShelfHz = highShelfHz;
        this.highShelfGainDb = highShelfGainDb;
        this.masterGainDb = masterGainDb;
        this.loudness = loudness;
        this.loudnessDriveDb = loudnessDriveDb;
        this.ceilingDb = ceilingDb;
    }

    public void process(short[] samples, int sampleCount, PcmAudioFormat format) {
        if (lowShelfGainDb != 0.0d) {
            new LowShelfEqualizerProcessor(lowShelfHz, lowShelfGainDb, 1.0d).process(samples, sampleCount, format);
        }
        if (midGainDb != 0.0d) {
            new ParametricEqualizerProcessor(midHz, midGainDb, midQ).process(samples, sampleCount, format);
        }
        if (highShelfGainDb != 0.0d) {
            new HighShelfEqualizerProcessor(highShelfHz, highShelfGainDb, 1.0d).process(samples, sampleCount, format);
        }
        if (loudness) {
            // Fold the master gain into the drive so the make-up gain is applied through the soft limiter
            // instead of being hard-clipped by a separate gain stage.
            saturate(samples, sampleCount, masterGainDb + loudnessDriveDb);
        } else if (masterGainDb != 0.0d) {
            new GainProcessor(masterGainDb).process(samples, sampleCount, format);
        }
    }

    /** Soft-saturate toward the ceiling: out = ceiling * tanh(drive * in / ceiling). Never hard clips. */
    private void saturate(short[] samples, int sampleCount, double driveDb) {
        double ceilingLinear = Math.pow(10.0d, clampCeilingDb(ceilingDb) / 20.0d) * FULL_SCALE;
        double drive = Math.pow(10.0d, driveDb / 20.0d);
        for (int i = 0; i < sampleCount; i++) {
            double x = samples[i] / ceilingLinear;
            double y = Math.tanh(drive * x) * ceilingLinear;
            samples[i] = clampToShort(y);
        }
    }

    private static double clampCeilingDb(double value) {
        if (Double.isNaN(value) || value > 0.0d) {
            return 0.0d;
        }
        return value < -24.0d ? -24.0d : value;
    }

    private static short clampToShort(double value) {
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
