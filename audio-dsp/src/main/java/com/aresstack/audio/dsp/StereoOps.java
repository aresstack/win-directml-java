package com.aresstack.audio.dsp;

/**
 * Mid/side stereo operations shared by the stereo-only blocks. The mid (sum) carries centred content such as
 * on-axis speech; the side (difference) carries laterally distributed content. Adjusting their gains lets the
 * Mid/Side Processor, Center Speech Extractor and Stereo Width Control emphasize the centre or change width.
 * All operate in place on interleaved stereo PCM16.
 */
public final class StereoOps {

    private StereoOps() {
    }

    /**
     * Apply mid/side gains in place. {@code monoCompatibility} caps the side gain at 1.0 so the result stays
     * mono-compatible (no exaggerated difference signal that would collapse or cancel when summed to mono).
     */
    public static void applyMidSide(short[] samples, int count, double midGain, double sideGain,
                                    boolean monoCompatibility) {
        double side = monoCompatibility && sideGain > 1.0d ? 1.0d : sideGain;
        int frames = count / 2;
        for (int f = 0; f < frames; f++) {
            int base = 2 * f;
            double left = samples[base];
            double right = samples[base + 1];
            double mid = 0.5d * (left + right) * midGain;
            double sid = 0.5d * (left - right) * side;
            samples[base] = clamp(mid + sid);
            samples[base + 1] = clamp(mid - sid);
        }
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
