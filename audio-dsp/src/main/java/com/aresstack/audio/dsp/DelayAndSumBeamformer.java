package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Delay-and-sum beamformer for a synchronized microphone array with known geometry. It computes the target
 * direction's arrival delay at each microphone, aligns the channels with fractional delays and sums them
 * into a single mono output emphasizing that direction. Requires a valid {@link MicrophoneArrayProfile}
 * whose microphone count matches the channel count — it never fabricates positions.
 */
public final class DelayAndSumBeamformer {

    private DelayAndSumBeamformer() {
    }

    /**
     * @param azimuthDeg   target azimuth in degrees (0 = +x, 90 = +y)
     * @param elevationDeg target elevation in degrees (0 = horizontal plane)
     * @param speedOfSoundMmPerS speed of sound in millimetres per second (e.g. 343000)
     * @param weights per-channel weights (null = unity); output is normalized by the total weight
     * @param outputGain linear output gain
     * @return a mono buffer beamformed toward the target direction
     */
    public static short[] beamform(short[] samples, int count, PcmAudioFormat format,
                                   MicrophoneArrayProfile array, double azimuthDeg, double elevationDeg,
                                   double speedOfSoundMmPerS, double[] weights, double outputGain) {
        int channels = format.getChannels();
        if (array == null || array.getMicrophoneCount() != channels || channels < 2) {
            throw new IllegalArgumentException("Beamforming needs a geometry whose microphone count matches "
                    + "the channel count.");
        }
        int rate = format.getSampleRateHz();
        double az = Math.toRadians(azimuthDeg);
        double el = Math.toRadians(elevationDeg);
        double ux = Math.cos(el) * Math.cos(az);
        double uy = Math.cos(el) * Math.sin(az);
        double uz = Math.sin(el);

        double[] proj = new double[channels];
        double minProj = Double.POSITIVE_INFINITY;
        for (int c = 0; c < channels; c++) {
            double[] p = array.position(c);
            proj[c] = p[0] * ux + p[1] * uy + p[2] * uz; // millimetres along the arrival direction
            minProj = Math.min(minProj, proj[c]);
        }
        double[] delays = new double[channels];
        for (int c = 0; c < channels; c++) {
            delays[c] = (proj[c] - minProj) / speedOfSoundMmPerS * rate; // samples, all >= 0
        }

        short[] aligned = samples.clone();
        ChannelAligner.applyDelays(aligned, count, channels, delays);

        int frames = count / channels;
        double totalWeight = 0.0d;
        for (int c = 0; c < channels; c++) {
            totalWeight += weights != null && c < weights.length ? Math.abs(weights[c]) : 1.0d;
        }
        double norm = totalWeight > 1.0e-9d ? totalWeight : channels;
        short[] out = new short[frames];
        for (int f = 0; f < frames; f++) {
            double acc = 0.0d;
            int base = f * channels;
            for (int c = 0; c < channels; c++) {
                double w = weights != null && c < weights.length ? weights[c] : 1.0d;
                acc += w * aligned[base + c];
            }
            out[f] = clamp(acc / norm * outputGain);
        }
        return out;
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
