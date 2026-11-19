package com.aresstack.audio.dsp;

/** Low-shelf EQ: broadly boosts or cuts frequencies below the cutoff, leaving highs largely unchanged. */
public final class LowShelfEqualizerProcessor extends BiquadProcessor {

    public LowShelfEqualizerProcessor(final double cutoffHz, final double gainDb, final double slope) {
        super(new BiquadDesign() {
            public BiquadCoefficients design(int sampleRateHz) {
                return BiquadCoefficients.lowShelf(sampleRateHz, cutoffHz, gainDb, slope);
            }
        });
    }
}
