package com.aresstack.audio.dsp;

/** One peaking (bell) EQ band. Stack several blocks for a multi-band equalizer. */
public final class ParametricEqualizerProcessor extends BiquadProcessor {

    public ParametricEqualizerProcessor(final double centerHz, final double gainDb, final double q) {
        super(new BiquadDesign() {
            public BiquadCoefficients design(int sampleRateHz) {
                return BiquadCoefficients.peaking(sampleRateHz, centerHz, gainDb, q);
            }
        });
    }
}
