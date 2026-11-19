package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/** Gain applies a pure level change with safe clamping and a bit-exact 0 dB pass-through. */
public class GainProcessorTest {

    private static final PcmAudioFormat MONO_16K = new PcmAudioFormat(16000, 1, 16);

    @Test
    public void zeroDbIsBitIdentical() {
        short[] samples = {0, 100, -100, 12345, -12345, 32767, -32768};
        short[] expected = samples.clone();
        new GainProcessor(0.0d).process(samples, samples.length, MONO_16K);
        assertArrayEquals(expected, samples);
    }

    @Test
    public void positiveGainRaisesAmplitude() {
        short[] samples = {1000, -1000, 2000, -2000};
        new GainProcessor(6.0d).process(samples, samples.length, MONO_16K);
        // +6 dB ≈ ×1.995
        assertTrue(samples[0] > 1900 && samples[0] < 2100);
        assertTrue(samples[1] < -1900 && samples[1] > -2100);
    }

    @Test
    public void negativeGainLowersAmplitude() {
        short[] samples = {1000, -1000};
        new GainProcessor(-6.0d).process(samples, samples.length, MONO_16K);
        // -6 dB ≈ ×0.501
        assertTrue(samples[0] > 480 && samples[0] < 520);
        assertTrue(samples[1] < -480 && samples[1] > -520);
    }

    @Test
    public void largePositiveGainClampsWithoutWrapAround() {
        short[] samples = {20000, -20000, 30000, -30000};
        new GainProcessor(24.0d).process(samples, samples.length, MONO_16K);
        // Every boosted sample must saturate to the rail with the correct sign, never wrap.
        assertTrue(samples[0] == Short.MAX_VALUE);
        assertTrue(samples[1] == Short.MIN_VALUE);
        assertTrue(samples[2] == Short.MAX_VALUE);
        assertTrue(samples[3] == Short.MIN_VALUE);
    }
}
