package com.aresstack.audio.dsp;

/**
 * Per-channel quality measures shared by the Best Channel Selector and the Channel Health Analyzer: level,
 * clipping, DC offset and a combined quality score. Pure and interleaved-PCM16 based.
 */
public final class ChannelDiagnostics {

    private static final double FULL_SCALE = 32768.0d;
    private static final int CLIP_THRESHOLD = 32000;

    private ChannelDiagnostics() {
    }

    public static double rms(short[] samples, int count, int channels, int channel) {
        int ch = Math.max(1, channels);
        int frames = count / ch;
        if (frames == 0) {
            return 0.0d;
        }
        double sum = 0.0d;
        for (int t = 0; t < frames; t++) {
            double v = samples[t * ch + channel];
            sum += v * v;
        }
        return Math.sqrt(sum / frames);
    }

    public static double clipFraction(short[] samples, int count, int channels, int channel) {
        int ch = Math.max(1, channels);
        int frames = count / ch;
        if (frames == 0) {
            return 0.0d;
        }
        int clipped = 0;
        for (int t = 0; t < frames; t++) {
            if (Math.abs(samples[t * ch + channel]) >= CLIP_THRESHOLD) {
                clipped++;
            }
        }
        return (double) clipped / frames;
    }

    public static double dcOffset(short[] samples, int count, int channels, int channel) {
        int ch = Math.max(1, channels);
        int frames = count / ch;
        if (frames == 0) {
            return 0.0d;
        }
        double sum = 0.0d;
        for (int t = 0; t < frames; t++) {
            sum += samples[t * ch + channel];
        }
        return sum / frames;
    }

    /** @return the index of the highest-quality channel (loud, little clipping, not silent). */
    public static int bestChannelIndex(short[] samples, int count, int channels, int preferred) {
        int ch = Math.max(1, channels);
        int best = preferred >= 0 && preferred < ch ? preferred : 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < ch; c++) {
            double score = rms(samples, count, channels, c) * (1.0d - clipFraction(samples, count, channels, c));
            if (score > bestScore) {
                bestScore = score;
                best = c;
            }
        }
        return best;
    }

    /** @return a human-readable summary of per-channel health issues, or "All channels healthy." */
    public static String describeHealth(short[] samples, int count, int channels) {
        int ch = Math.max(1, channels);
        StringBuilder sb = new StringBuilder();
        for (int c = 0; c < ch; c++) {
            double rms = rms(samples, count, channels, c);
            double levelDb = 20.0d * Math.log10(Math.max(rms, 1.0e-9d) / FULL_SCALE);
            double clip = clipFraction(samples, count, channels, c);
            double dc = Math.abs(dcOffset(samples, count, channels, c)) / FULL_SCALE;
            if (rms < 1.0d) {
                append(sb, "channel " + c + " is silent");
            }
            if (clip > 0.01d) {
                append(sb, "channel " + c + " clips (" + Math.round(clip * 100) + "%)");
            }
            if (dc > 0.02d) {
                append(sb, "channel " + c + " has a DC offset");
            }
            if (levelDb < -50.0d && rms >= 1.0d) {
                append(sb, "channel " + c + " is very quiet");
            }
        }
        return sb.length() == 0 ? "All channels healthy." : sb.toString();
    }

    private static void append(StringBuilder sb, String issue) {
        if (sb.length() > 0) {
            sb.append("; ");
        }
        sb.append(issue);
    }
}
