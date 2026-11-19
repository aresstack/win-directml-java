package com.aresstack.audio.dsp;

/** High-shelf EQ: broadly boosts or cuts frequencies above the cutoff, leaving lows largely unchanged. */
public final class HighShelfEqualizerProcessor extends BiquadProcessor {

    public HighShelfEqualizerProcessor(final double cutoffHz, final double gainDb, final double slope) {
        super(new BiquadDesign() {
            public BiquadCoefficients design(int sampleRateHz) {
                return BiquadCoefficients.highShelf(sampleRateHz, cutoffHz, gainDb, slope);
            }
        });
    }
}
