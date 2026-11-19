package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/** The de-esser reduces a loud sibilance-band tone while leaving other frequencies and quiet input alone. */
public class DeEsserProcessorTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);

    @Test
    public void reducesALoudSibilanceBandTone() {
        short[] input = tone(6500.0d, RATE, 8000);
        short[] out = process(new DeEsserSettings(6500.0d, 2500.0d, -40.0d, 15.0d, 2.0d, 60.0d), input);
        assertTrue("sibilance band should be reduced", rms(out) < 0.8d * rms(input));
    }

    @Test
    public void leavesFrequenciesOutsideTheBandLargelyUntouched() {
        short[] input = tone(300.0d, RATE, 8000);
        short[] out = process(new DeEsserSettings(6500.0d, 2000.0d, -40.0d, 15.0d, 2.0d, 60.0d), input);
        assertTrue("a low tone is outside the sibilance band", rms(out) > 0.9d * rms(input));
    }

    @Test
    public void doesNotReduceBelowThreshold() {
        short[] input = tone(6500.0d, RATE, 60); // very quiet
        short[] out = process(new DeEsserSettings(6500.0d, 2500.0d, -20.0d, 15.0d, 2.0d, 60.0d), input);
        assertTrue("below threshold there is no reduction", rms(out) > 0.9d * rms(input));
    }

    @Test
    public void silenceStaysSilent() {
        short[] out = process(new DeEsserSettings(6500.0d, 2500.0d, -40.0d, 15.0d, 2.0d, 60.0d), new short[RATE]);
        for (short sample : out) {
            assertTrue(sample == 0);
        }
    }

    private static short[] process(DeEsserSettings s, short[] input) {
        short[] copy = input.clone();
        new DeEsserProcessor(s).process(copy, copy.length, MONO);
        return copy;
    }

    private static short[] tone(double freq, int n, int amp) {
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            out[i] = (short) Math.round(amp * Math.sin(2.0d * Math.PI * freq * i / RATE));
        }
        return out;
    }

    private static double rms(short[] samples) {
        long sum = 0;
        int from = samples.length / 2;
        for (int i = from; i < samples.length; i++) {
            sum += (long) samples[i] * samples[i];
        }
        return Math.sqrt((double) sum / (samples.length - from));
    }
}
