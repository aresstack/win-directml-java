package com.aresstack.audio.dsp;

/** Per-channel delay memory for a transposed Direct-Form-II biquad. Reset to clear any tail. */
final class BiquadFilterState {

    double z1;
    double z2;

    void reset() {
        z1 = 0.0d;
        z2 = 0.0d;
    }

    /** Advance the transposed Direct-Form-II filter by one sample and return the filtered value. */
    double process(BiquadCoefficients c, double x) {
        double y = c.b0 * x + z1;
        z1 = c.b1 * x - c.a1 * y + z2;
        z2 = c.b2 * x - c.a2 * y;
        return y;
    }
}
