package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/** Plosive reduction ducks low-frequency transient bursts but leaves steady low tones and highs alone. */
public class PlosiveReductionProcessorTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);

    @Test
    public void ducksALowFrequencyTransientBurst() {
        short[] input = burst();
        double reduced = burstEnergy(process(new PlosiveReductionSettings(0.9d, 120.0d, 3.0d, 80.0d), input));
        double original = burstEnergy(process(new PlosiveReductionSettings(0.0d, 120.0d, 3.0d, 80.0d), input));
        assertTrue("the plosive burst should be attenuated: reduced=" + reduced + " original=" + original,
                reduced < 0.8d * original);
    }

    @Test
    public void leavesASteadyLowToneLargelyUntouched() {
        short[] input = tone(60.0d, RATE, 6000);
        short[] out = process(new PlosiveReductionSettings(0.9d, 120.0d, 3.0d, 80.0d), input);
        assertTrue("a steady low tone is not a transient", rms(out) > 0.85d * rms(input));
    }

    @Test
    public void leavesHighFrequencyContentUntouched() {
        short[] input = tone(4000.0d, RATE, 6000);
        short[] out = process(new PlosiveReductionSettings(0.9d, 120.0d, 3.0d, 80.0d), input);
        assertTrue("high content is above the plosive band", rms(out) > 0.95d * rms(input));
    }

    @Test
    public void silenceStaysStable() {
        short[] out = process(new PlosiveReductionSettings(0.9d, 120.0d, 3.0d, 80.0d), new short[RATE]);
        for (short sample : out) {
            assertTrue(sample == 0);
        }
    }

    private static short[] burst() {
        // 0.3 s near-silence, a 30 ms 60 Hz burst, then silence.
        short[] lead = new short[RATE * 3 / 10];
        short[] pop = tone(60.0d, RATE * 30 / 1000, 12000);
        short[] tail = new short[RATE / 2];
        short[] out = new short[lead.length + pop.length + tail.length];
        System.arraycopy(lead, 0, out, 0, lead.length);
        System.arraycopy(pop, 0, out, lead.length, pop.length);
        return out;
    }

    private static double burstEnergy(short[] samples) {
        int from = RATE * 3 / 10;
        int to = from + RATE * 30 / 1000;
        long sum = 0;
        for (int i = from; i < to; i++) {
            sum += (long) samples[i] * samples[i];
        }
        return Math.sqrt((double) sum / (to - from));
    }

    private static short[] process(PlosiveReductionSettings s, short[] input) {
        short[] copy = input.clone();
        new PlosiveReductionProcessor(s).process(copy, copy.length, MONO);
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
