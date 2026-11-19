package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

/**
 * Compute the retained sample range for a leading/trailing silence trim, driven purely by an upstream
 * {@link SpeechActivityTrack} (no hidden energy detector). Only silence before the first and after the last
 * detected speech is removed — inner pauses are always kept, so the timing and later timestamp mapping of
 * the speech stay intact. The caller cuts the buffer to the returned range and shifts/invalidates metadata.
 */
public final class SilenceTrimmer {

    /** The retained interleaved sample range plus flags describing what was decided. */
    public static final class TrimBounds {
        public final int startInterleaved;
        public final int endInterleaved;
        public final boolean trimmed;
        public final boolean noSpeech;

        TrimBounds(int startInterleaved, int endInterleaved, boolean trimmed, boolean noSpeech) {
            this.startInterleaved = startInterleaved;
            this.endInterleaved = endInterleaved;
            this.trimmed = trimmed;
            this.noSpeech = noSpeech;
        }
    }

    public TrimBounds computeBounds(short[] samples, PcmAudioFormat format, SpeechActivityTrack track,
                                    SilenceTrimmerSettings settings) {
        int channels = Math.max(1, format.getChannels());
        int rate = format.getSampleRateHz();
        int total = samples.length - samples.length % channels;
        int frameInterleaved = track == null ? 0 : track.getFrameSampleCountInterleaved();
        if (frameInterleaved <= 0 || track.size() == 0) {
            return new TrimBounds(0, total, false, false); // no usable track: caller keeps the signal
        }

        int firstSpeech = -1;
        int lastSpeech = -1;
        for (int i = 0; i < track.size(); i++) {
            if (track.getFrames().get(i).getSpeechProbability() >= settings.getMinSpeechProbability()) {
                if (firstSpeech < 0) {
                    firstSpeech = i;
                }
                lastSpeech = i;
            }
        }
        if (firstSpeech < 0) {
            return new TrimBounds(0, total, false, true); // no speech at all
        }

        int preRoll = align(msToSamples(settings.getPreRollMs(), rate) * channels, channels);
        int postRoll = align(msToSamples(settings.getPostRollMs(), rate) * channels, channels);
        int start = settings.isTrimLeading() ? firstSpeech * frameInterleaved - preRoll : 0;
        int end = settings.isTrimTrailing() ? (lastSpeech + 1) * frameInterleaved + postRoll : total;

        start = align(clampIndex(start, 0, total), channels);
        end = align(clampIndex(end, 0, total), channels);
        if (end <= start) {
            return new TrimBounds(0, total, false, false);
        }

        int[] widened = enforceMinimumRetained(start, end, total, channels,
                align(msToSamples(settings.getMinRetainedMs(), rate) * channels, channels));
        start = widened[0];
        end = widened[1];

        if (settings.isZeroCrossingAlignment()) {
            int window = align(msToSamples(settings.getZeroCrossingSearchMs(), rate) * channels, channels);
            if (settings.isTrimLeading() && start > 0) {
                start = alignToZeroCrossing(samples, channels, start, window, 0, end);
            }
            if (settings.isTrimTrailing() && end < total) {
                end = alignToZeroCrossing(samples, channels, end, window, start, total);
            }
        }

        boolean trimmed = start > 0 || end < total;
        return new TrimBounds(start, end, trimmed, false);
    }

    private static int[] enforceMinimumRetained(int start, int end, int total, int channels, int minRetained) {
        if (minRetained <= 0 || end - start >= minRetained || total < minRetained) {
            return new int[]{start, end};
        }
        int deficit = minRetained - (end - start);
        int extendEnd = Math.min(deficit, total - end);
        end += extendEnd;
        deficit -= extendEnd;
        if (deficit > 0) {
            start = Math.max(0, start - deficit);
        }
        return new int[]{align(start, channels), align(end, channels)};
    }

    /** Find the nearest zero crossing (channel-0 sign change) within {@code window} of {@code target}. */
    private static int alignToZeroCrossing(short[] samples, int channels, int target, int window,
                                           int lowerBound, int upperBound) {
        for (int offset = 0; offset <= window; offset += channels) {
            int forward = target + offset;
            if (forward > lowerBound && forward < upperBound && isCrossing(samples, channels, forward)) {
                return forward;
            }
            int backward = target - offset;
            if (backward > lowerBound && backward < upperBound && isCrossing(samples, channels, backward)) {
                return backward;
            }
        }
        return target;
    }

    private static boolean isCrossing(short[] samples, int channels, int interleavedIndex) {
        int current = samples[interleavedIndex];
        int previous = samples[interleavedIndex - channels];
        return (current >= 0) != (previous >= 0);
    }

    private static int msToSamples(double milliseconds, int rate) {
        return (int) Math.round(milliseconds * rate / 1000.0d);
    }

    private static int align(int value, int channels) {
        return value - value % channels;
    }

    private static int clampIndex(int value, int min, int max) {
        if (value < min) {
            return min;
        }
        return value > max ? max : value;
    }
}
