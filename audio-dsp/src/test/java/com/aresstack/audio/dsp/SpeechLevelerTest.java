package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** The Speech Leveler raises quiet speech, tames loud speech and does not lift silence. */
public class SpeechLevelerTest {

    private static final int RATE = 16000;
    private static final int FRAME = 160;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);

    private static SpeechLevelerSettings defaults() {
        return new SpeechLevelerSettings(-20.0d, 18.0d, 12.0d, 100.0d, 800.0d, 300.0d,
                12.0d, 0.5d, 6.0d, true, true);
    }

    @Test
    public void raisesQuietSpeechTowardTheTargetLevel() {
        short[] signal = tone(300.0d, 48000, 1000); // ~-33 dBFS RMS
        SpeechActivityTrack track = track(signal.length, true);
        new SpeechLevelerProcessor(defaults()).process(signal, signal.length, MONO,
                new SpeechLevelerState(), track);
        double outDb = dbfs(rms(signal, 36000, 48000));
        assertTrue("quiet speech raised toward target: " + outDb + " dBFS", outDb > -26.0d && outDb < -15.0d);
    }

    @Test
    public void tamesLoudSpeechTowardTheTargetLevel() {
        short[] signal = tone(300.0d, 48000, 20000); // ~-7 dBFS RMS
        double inDb = dbfs(rms(signal, 36000, 48000));
        SpeechActivityTrack track = track(signal.length, true);
        new SpeechLevelerProcessor(defaults()).process(signal, signal.length, MONO,
                new SpeechLevelerState(), track);
        double outDb = dbfs(rms(signal, 36000, 48000));
        assertTrue("loud speech reduced: " + inDb + " -> " + outDb, outDb < inDb - 6.0d);
        assertTrue("not over-reduced: " + outDb, outDb > -24.0d);
    }

    @Test
    public void doesNotLiftDetectedSilence() {
        short[] signal = noise(24000, 500); // quiet background, all marked non-speech
        double inRms = rms(signal, 0, signal.length);
        short[] work = signal.clone();
        new SpeechLevelerProcessor(defaults()).process(work, work.length, MONO,
                new SpeechLevelerState(), track(work.length, false));
        double outRms = rms(work, 0, work.length);
        assertTrue("silence not amplified: " + inRms + " -> " + outRms, outRms <= inRms * 1.2d);
    }

    @Test
    public void reproducibleForIdenticalInputAndSettings() {
        short[] a = tone(300.0d, 16000, 3000);
        short[] b = a.clone();
        SpeechActivityTrack track = track(a.length, true);
        new SpeechLevelerProcessor(defaults()).process(a, a.length, MONO, new SpeechLevelerState(), track);
        new SpeechLevelerProcessor(defaults()).process(b, b.length, MONO, new SpeechLevelerState(),
                track(b.length, true));
        assertEquals(a.length, b.length);
        for (int i = 0; i < a.length; i++) {
            assertEquals("sample " + i, a[i], b[i]);
        }
    }

    private static SpeechActivityTrack track(int length, boolean speech) {
        SpeechActivityTrack track = new SpeechActivityTrack(RATE, 1, FRAME);
        int frames = length / FRAME + 1;
        for (int i = 0; i < frames; i++) {
            track.add(new SpeechActivityMetadata(speech ? 1.0d : 0.0d, speech, -60.0d, -20.0d));
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

    private static short[] noise(int n, int amp) {
        short[] out = new short[n];
        int state = 777;
        for (int i = 0; i < n; i++) {
            state = state * 1103515245 + 12345;
            out[i] = (short) (((state >> 16) % (2 * amp)) - amp);
        }
        return out;
    }

    private static double rms(short[] s, int from, int to) {
        long sum = 0;
        for (int i = from; i < to; i++) {
            sum += (long) s[i] * s[i];
        }
        return Math.sqrt((double) sum / (to - from));
    }

    private static double dbfs(double rms) {
        return 20.0d * Math.log10(Math.max(rms, 1.0e-9d) / 32768.0d);
    }
}
