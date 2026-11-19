package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/** Breath reduction attenuates audible non-speech using the VAD track and protects speech. */
public class BreathReductionProcessorTest {

    private static final int RATE = 16000;
    private static final int FRAME = 320;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);

    @Test
    public void attenuatesAudibleNonSpeechButProtectsSpeech() {
        // Frames 0..19 speech, 25..44 non-speech breath; both audible.
        short[] input = tone(500.0d, 60 * FRAME, 4000);
        SpeechActivityTrack track = track(60, 0, 19);
        short[] out = process(new BreathReductionSettings(0.7d, 18.0d, true, 5.0d, 120.0d), input, track);

        double speech = rms(out, 5 * FRAME, 15 * FRAME) / rms(input, 5 * FRAME, 15 * FRAME);
        double breath = rms(out, 30 * FRAME, 44 * FRAME) / rms(input, 30 * FRAME, 44 * FRAME);
        assertTrue("speech is protected", speech > 0.9d);
        assertTrue("breath is attenuated", breath < 0.6d);
    }

    @Test
    public void withoutATrackItPassesThrough() {
        short[] input = tone(500.0d, RATE, 4000);
        short[] out = process(new BreathReductionSettings(0.7d, 18.0d, true, 5.0d, 120.0d), input, null);
        assertArrayEquals(input, out);
    }

    private static short[] process(BreathReductionSettings s, short[] input, SpeechActivityTrack track) {
        short[] copy = input.clone();
        new BreathReductionProcessor(s).process(copy, copy.length, MONO, track);
        return copy;
    }

    private static SpeechActivityTrack track(int frames, int speechFrom, int speechTo) {
        SpeechActivityTrack track = new SpeechActivityTrack(RATE, 1, FRAME);
        for (int f = 0; f < frames; f++) {
            boolean speech = f >= speechFrom && f <= speechTo;
            track.add(new SpeechActivityMetadata(speech ? 0.9d : 0.1d, speech, -60.0d, -20.0d));
        }
        return track;
    }

    private static short[] tone(double freq, int n, int amp) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amp * Math.sin(2.0d * Math.PI * freq * i / RATE));
        }
        return out;
    }

    private static double rms(short[] samples, int from, int to) {
        long sum = 0;
        for (int i = from; i < to; i++) {
            sum += (long) samples[i] * samples[i];
        }
        return Math.sqrt((double) sum / (to - from));
    }
}
