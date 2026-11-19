package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Leading/trailing silence trimming driven purely by the speech-activity track; inner pauses are kept. */
public class SilenceTrimmerTest {

    private static final int RATE = 16000;
    private static final int FRAME = 320; // 20 ms
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);

    private final SilenceTrimmer trimmer = new SilenceTrimmer();

    @Test
    public void removesLeadingAndTrailingSilenceButKeepsInnerPause() {
        // speech in frames 10..12 and 18..20; silence elsewhere (incl. the inner pause 13..17).
        SpeechActivityTrack track = track(40, new int[]{10, 12}, new int[]{18, 20});
        short[] samples = new short[40 * FRAME];
        SilenceTrimmer.TrimBounds bounds = trimmer.computeBounds(samples, MONO, track, settings(true, true, 0, 0));

        assertTrue(bounds.trimmed);
        assertEquals(10 * FRAME, bounds.startInterleaved);
        assertEquals(21 * FRAME, bounds.endInterleaved);
        // The inner pause (frames 13..17) lies inside the retained range → preserved.
        assertTrue(bounds.startInterleaved <= 13 * FRAME && bounds.endInterleaved >= 18 * FRAME);
    }

    @Test
    public void preRollAndPostRollAreKept() {
        SpeechActivityTrack track = track(40, new int[]{10, 20});
        short[] samples = new short[40 * FRAME];
        int preRollMs = 100;  // 1600 samples
        int postRollMs = 100;
        SilenceTrimmer.TrimBounds bounds = trimmer.computeBounds(samples, MONO, track,
                settings(true, true, preRollMs, postRollMs));
        assertEquals(10 * FRAME - 1600, bounds.startInterleaved);
        assertEquals(21 * FRAME + 1600, bounds.endInterleaved);
    }

    @Test
    public void onlyLeadingDoesNotTrimTheEnd() {
        SpeechActivityTrack track = track(40, new int[]{10, 20});
        short[] samples = new short[40 * FRAME];
        SilenceTrimmer.TrimBounds bounds = trimmer.computeBounds(samples, MONO, track, settings(true, false, 0, 0));
        assertEquals(10 * FRAME, bounds.startInterleaved);
        assertEquals(samples.length, bounds.endInterleaved);
    }

    @Test
    public void onlyTrailingDoesNotTrimTheStart() {
        SpeechActivityTrack track = track(40, new int[]{10, 20});
        short[] samples = new short[40 * FRAME];
        SilenceTrimmer.TrimBounds bounds = trimmer.computeBounds(samples, MONO, track, settings(false, true, 0, 0));
        assertEquals(0, bounds.startInterleaved);
        assertEquals(21 * FRAME, bounds.endInterleaved);
    }

    @Test
    public void noSpeechIsFlaggedAndNothingIsTrimmed() {
        SpeechActivityTrack track = track(20); // no speech frames
        short[] samples = new short[20 * FRAME];
        SilenceTrimmer.TrimBounds bounds = trimmer.computeBounds(samples, MONO, track, settings(true, true, 0, 0));
        assertTrue(bounds.noSpeech);
        assertFalse(bounds.trimmed);
        assertEquals(0, bounds.startInterleaved);
        assertEquals(samples.length, bounds.endInterleaved);
    }

    @Test
    public void boundsStayWithinTheBuffer() {
        SpeechActivityTrack track = track(10, new int[]{0, 9}); // speech everywhere, huge roll
        short[] samples = new short[10 * FRAME];
        SilenceTrimmer.TrimBounds bounds = trimmer.computeBounds(samples, MONO, track, settings(true, true, 5000, 5000));
        assertTrue(bounds.startInterleaved >= 0);
        assertTrue(bounds.endInterleaved <= samples.length);
        assertTrue(bounds.startInterleaved < bounds.endInterleaved);
    }

    @Test
    public void zeroCrossingSearchStaysWithinItsWindow() {
        SpeechActivityTrack track = track(40, new int[]{10, 20});
        short[] samples = new short[40 * FRAME];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (short) Math.round(6000.0d * Math.sin(2.0d * Math.PI * 200.0d * i / RATE));
        }
        int searchMs = 5; // 80 samples
        SilenceTrimmerSettings s = new SilenceTrimmerSettings(true, true, 0.5d, 0, 0, 0,
                SilenceTrimNoSpeechBehavior.KEEP_ORIGINAL, true, searchMs);
        SilenceTrimmer.TrimBounds withZc = trimmer.computeBounds(samples, MONO, track, s);
        assertTrue("zero-crossing alignment must not move the start past its window",
                Math.abs(withZc.startInterleaved - 10 * FRAME) <= 80);
    }

    // ------------------------------------------------------------------ helpers

    private static SilenceTrimmerSettings settings(boolean lead, boolean trail, int preMs, int postMs) {
        return new SilenceTrimmerSettings(lead, trail, 0.5d, preMs, postMs, 0,
                SilenceTrimNoSpeechBehavior.KEEP_ORIGINAL, false, 5.0d);
    }

    /** Build a track of {@code frames} frames; each int[] {from,to} marks an inclusive speech range. */
    private static SpeechActivityTrack track(int frames, int[]... speechRanges) {
        SpeechActivityTrack track = new SpeechActivityTrack(RATE, 1, FRAME);
        for (int f = 0; f < frames; f++) {
            boolean speech = false;
            for (int[] range : speechRanges) {
                if (f >= range[0] && f <= range[1]) {
                    speech = true;
                }
            }
            track.add(new SpeechActivityMetadata(speech ? 0.9d : 0.0d, speech, -60.0d, speech ? -12.0d : -80.0d));
        }
        return track;
    }
}
