package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import java.util.List;

/**
 * Mute non-speech regions of a recording using an upstream {@link SpeechActivityTrack}. It contains no
 * speech detector of its own: the open/closed decision comes entirely from the supplied track. Non-speech
 * samples outside the protected boundaries are set to exact digital silence ({@code 0}); pre-/post-roll
 * extend the open region around speech, and short open/close fades only shape the transitions — the stable
 * closed region stays exactly zero. All channels open and close together on the same decision, and the
 * sample count and duration are left unchanged.
 */
public final class SpeechActivityGate {

    private SpeechActivityGate() {
    }

    public static void apply(short[] samples, int count, PcmAudioFormat format, SpeechActivityTrack track,
                             double minSpeechProbability, double preRollMs, double postRollMs,
                             double openFadeMs, double closeFadeMs) {
        int channels = Math.max(1, format.getChannels());
        int frameSamples = track.getFrameSampleCountPerChannel();
        List<SpeechActivityMetadata> frames = track.getFrames();
        int frameCount = frames.size();
        if (frameSamples <= 0 || frameCount == 0) {
            return;
        }
        double frameMillis = 1000.0d * frameSamples / track.getSampleRateHz();
        int preRollFrames = framesForMillis(preRollMs, frameMillis);
        int postRollFrames = framesForMillis(postRollMs, frameMillis);

        boolean[] open = new boolean[frameCount];
        for (int i = 0; i < frameCount; i++) {
            SpeechActivityMetadata frame = frames.get(i);
            if (!frame.isSpeechActive() && frame.getSpeechProbability() < minSpeechProbability) {
                continue;
            }
            int first = Math.max(0, i - preRollFrames);
            int last = Math.min(frameCount - 1, i + postRollFrames);
            for (int f = first; f <= last; f++) {
                open[f] = true;
            }
        }

        int totalFrames = count / channels;
        int attackSamples = millisToSamples(track.getSampleRateHz(), openFadeMs);
        int releaseSamples = millisToSamples(track.getSampleRateHz(), closeFadeMs);
        double gain = open[0] ? 1.0d : 0.0d;
        for (int sf = 0; sf < totalFrames; sf++) {
            int activityFrame = Math.min(frameCount - 1, sf / frameSamples);
            boolean isOpen = open[activityFrame];
            gain = isOpen ? approach(gain, 1.0d, attackSamples) : approach(gain, 0.0d, releaseSamples);
            int base = sf * channels;
            if (gain <= 0.0d) {
                for (int c = 0; c < channels; c++) {
                    samples[base + c] = 0; // exact digital silence in the stable closed region
                }
            } else if (gain < 1.0d) {
                for (int c = 0; c < channels; c++) {
                    samples[base + c] = clampToShort(samples[base + c] * gain);
                }
            }
            // gain == 1.0: leave the samples untouched so open regions stay bit-identical.
        }
    }

    private static int framesForMillis(double millis, double frameMillis) {
        if (!finiteNonNegative(millis) || frameMillis <= 0.0d) {
            return 0;
        }
        return (int) Math.ceil(millis / frameMillis);
    }

    private static double approach(double current, double target, int samples) {
        if (samples <= 0) {
            return target;
        }
        double step = 1.0d / samples;
        if (target > current) {
            return Math.min(target, current + step);
        }
        return Math.max(target, current - step);
    }

    private static int millisToSamples(int sampleRate, double millis) {
        if (!finiteNonNegative(millis)) {
            return 0;
        }
        return Math.max(0, (int) Math.round(sampleRate * millis / 1000.0d));
    }

    private static short clampToShort(double value) {
        return (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, Math.round(value)));
    }

    private static boolean finiteNonNegative(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value) && value >= 0.0d;
    }
}
