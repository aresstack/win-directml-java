package com.aresstack.audio.dsp;

/**
 * Normalized biquad coefficients (a0 folded to 1), computed with the standard Audio-EQ-Cookbook (RBJ)
 * formulas for peaking and shelving filters. Factories validate their arguments and always return finite,
 * stable coefficients; invalid input throws {@link IllegalArgumentException} (callers bypass rather than
 * feed unstable coefficients into the signal path).
 */
public final class BiquadCoefficients {

    public final double b0;
    public final double b1;
    public final double b2;
    public final double a1;
    public final double a2;

    private BiquadCoefficients(double b0, double b1, double b2, double a1, double a2) {
        if (!finite(b0) || !finite(b1) || !finite(b2) || !finite(a1) || !finite(a2)) {
            throw new IllegalArgumentException("Biquad coefficients must be finite.");
        }
        this.b0 = b0;
        this.b1 = b1;
        this.b2 = b2;
        this.a1 = a1;
        this.a2 = a2;
    }

    /** Band-pass with constant 0 dB peak gain (unity at the center) — used to isolate a band for analysis. */
    public static BiquadCoefficients bandPass(int sampleRateHz, double centerHz, double q) {
        requireFrequency(sampleRateHz, centerHz);
        requireQ(q);
        double w0 = omega(sampleRateHz, centerHz);
        double cosw0 = Math.cos(w0);
        double alpha = Math.sin(w0) / (2.0d * q);
        return normalized(alpha, 0.0d, -alpha, 1.0d + alpha, -2.0d * cosw0, 1.0d - alpha);
    }

    /** Second-order low-pass, used to split off a low band for transient/plosive handling. */
    public static BiquadCoefficients lowPass(int sampleRateHz, double cutoffHz, double q) {
        requireFrequency(sampleRateHz, cutoffHz);
        requireQ(q);
        double w0 = omega(sampleRateHz, cutoffHz);
        double cosw0 = Math.cos(w0);
        double alpha = Math.sin(w0) / (2.0d * q);
        double oneMinusCos = 1.0d - cosw0;
        return normalized(oneMinusCos / 2.0d, oneMinusCos, oneMinusCos / 2.0d,
                1.0d + alpha, -2.0d * cosw0, 1.0d - alpha);
    }

    private static void requireQ(double q) {
        if (!(q > 0.0d) || !finite(q)) {
            throw new IllegalArgumentException("Q must be > 0.");
        }
    }

    /** Peaking (bell) EQ: boosts/cuts around {@code centerHz} with bandwidth set by {@code q}. */
    public static BiquadCoefficients peaking(int sampleRateHz, double centerHz, double gainDb, double q) {
        requireFrequency(sampleRateHz, centerHz);
        if (!(q > 0.0d) || !finite(q)) {
            throw new IllegalArgumentException("Q must be > 0.");
        }
        requireGain(gainDb);
        double a = Math.pow(10.0d, gainDb / 40.0d);
        double w0 = omega(sampleRateHz, centerHz);
        double cosw0 = Math.cos(w0);
        double alpha = Math.sin(w0) / (2.0d * q);
        double b0 = 1.0d + alpha * a;
        double b1 = -2.0d * cosw0;
        double b2 = 1.0d - alpha * a;
        double a0 = 1.0d + alpha / a;
        double a1 = -2.0d * cosw0;
        double a2 = 1.0d - alpha / a;
        return normalized(b0, b1, b2, a0, a1, a2);
    }

    /** Low-shelf EQ: boosts/cuts frequencies below {@code cutoffHz}; {@code slope} shapes the transition. */
    public static BiquadCoefficients lowShelf(int sampleRateHz, double cutoffHz, double gainDb, double slope) {
        requireFrequency(sampleRateHz, cutoffHz);
        requireGain(gainDb);
        double a = Math.pow(10.0d, gainDb / 40.0d);
        double w0 = omega(sampleRateHz, cutoffHz);
        double cosw0 = Math.cos(w0);
        double alpha = shelfAlpha(w0, a, slope);
        double twoSqrtAalpha = 2.0d * Math.sqrt(a) * alpha;
        double b0 = a * ((a + 1.0d) - (a - 1.0d) * cosw0 + twoSqrtAalpha);
        double b1 = 2.0d * a * ((a - 1.0d) - (a + 1.0d) * cosw0);
        double b2 = a * ((a + 1.0d) - (a - 1.0d) * cosw0 - twoSqrtAalpha);
        double a0 = (a + 1.0d) + (a - 1.0d) * cosw0 + twoSqrtAalpha;
        double a1 = -2.0d * ((a - 1.0d) + (a + 1.0d) * cosw0);
        double a2 = (a + 1.0d) + (a - 1.0d) * cosw0 - twoSqrtAalpha;
        return normalized(b0, b1, b2, a0, a1, a2);
    }

    /** High-shelf EQ: boosts/cuts frequencies above {@code cutoffHz}; {@code slope} shapes the transition. */
    public static BiquadCoefficients highShelf(int sampleRateHz, double cutoffHz, double gainDb, double slope) {
        requireFrequency(sampleRateHz, cutoffHz);
        requireGain(gainDb);
        double a = Math.pow(10.0d, gainDb / 40.0d);
        double w0 = omega(sampleRateHz, cutoffHz);
        double cosw0 = Math.cos(w0);
        double alpha = shelfAlpha(w0, a, slope);
        double twoSqrtAalpha = 2.0d * Math.sqrt(a) * alpha;
        double b0 = a * ((a + 1.0d) + (a - 1.0d) * cosw0 + twoSqrtAalpha);
        double b1 = -2.0d * a * ((a - 1.0d) + (a + 1.0d) * cosw0);
        double b2 = a * ((a + 1.0d) + (a - 1.0d) * cosw0 - twoSqrtAalpha);
        double a0 = (a + 1.0d) - (a - 1.0d) * cosw0 + twoSqrtAalpha;
        double a1 = 2.0d * ((a - 1.0d) - (a + 1.0d) * cosw0);
        double a2 = (a + 1.0d) - (a - 1.0d) * cosw0 - twoSqrtAalpha;
        return normalized(b0, b1, b2, a0, a1, a2);
    }

    private static double shelfAlpha(double w0, double a, double slope) {
        if (!(slope > 0.0d) || !finite(slope)) {
            throw new IllegalArgumentException("Shelf slope must be > 0.");
        }
        double term = (a + 1.0d / a) * (1.0d / slope - 1.0d) + 2.0d;
        if (term < 0.0d) {
            term = 0.0d; // guard against tiny negative rounding under extreme slope/gain
        }
        return Math.sin(w0) / 2.0d * Math.sqrt(term);
    }

    private static BiquadCoefficients normalized(double b0, double b1, double b2,
                                                 double a0, double a1, double a2) {
        if (a0 == 0.0d || !finite(a0)) {
            throw new IllegalArgumentException("Unstable biquad (a0 = " + a0 + ").");
        }
        return new BiquadCoefficients(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
    }

    private static double omega(int sampleRateHz, double frequencyHz) {
        return 2.0d * Math.PI * frequencyHz / sampleRateHz;
    }

    private static void requireFrequency(int sampleRateHz, double frequencyHz) {
        if (sampleRateHz <= 0) {
            throw new IllegalArgumentException("Sample rate must be positive.");
        }
        if (!(frequencyHz > 0.0d) || !finite(frequencyHz)) {
            throw new IllegalArgumentException("Frequency must be > 0.");
        }
        if (frequencyHz >= sampleRateHz / 2.0d) {
            throw new IllegalArgumentException("Frequency " + frequencyHz
                    + " Hz must stay below Nyquist " + (sampleRateHz / 2.0d) + " Hz.");
        }
    }

    private static void requireGain(double gainDb) {
        if (!finite(gainDb)) {
            throw new IllegalArgumentException("Gain must be a finite dB value.");
        }
    }

    private static boolean finite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }
}
