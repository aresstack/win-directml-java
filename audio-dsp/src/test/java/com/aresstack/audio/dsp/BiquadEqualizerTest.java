package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Peaking and shelving equalizers behave as designed and stay numerically stable. */
public class BiquadEqualizerTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);
    private static final int N = RATE; // one second

    // ------------------------------------------------------------------ parametric (peaking)

    @Test
    public void peakingBoostRaisesEnergyNearCenter() {
        short[] input = sine(1000.0d, 6000);
        double ratio = rmsRatioAfter(new ParametricEqualizerProcessor(1000.0d, 12.0d, 1.0d), input);
        assertTrue("expected a clear boost near the center, got " + ratio, ratio > 2.0d);
    }

    @Test
    public void peakingCutLowersEnergyNearCenter() {
        short[] input = sine(1000.0d, 6000);
        double ratio = rmsRatioAfter(new ParametricEqualizerProcessor(1000.0d, -12.0d, 1.0d), input);
        assertTrue("expected a clear cut near the center, got " + ratio, ratio < 0.6d);
    }

    @Test
    public void peakingBarelyAffectsDistantFrequencies() {
        short[] input = sine(6000.0d, 6000); // far above a 300 Hz band
        double ratio = rmsRatioAfter(new ParametricEqualizerProcessor(300.0d, 12.0d, 4.0d), input);
        assertTrue("distant frequency should be nearly unchanged, got " + ratio, ratio > 0.85d && ratio < 1.2d);
    }

    @Test
    public void peakingZeroDbIsTransparent() {
        short[] input = sine(1200.0d, 8000);
        short[] output = process(new ParametricEqualizerProcessor(1200.0d, 0.0d, 1.0d), input);
        int maxDiff = 0;
        for (int i = 0; i < input.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(input[i] - output[i]));
        }
        assertTrue("0 dB peaking must be transparent, max diff " + maxDiff, maxDiff <= 2);
    }

    @Test
    public void lowerQGivesAWiderBandThanHigherQ() {
        short[] offCenter = sine(1600.0d, 6000); // off the 1000 Hz center
        double wide = rmsRatioAfter(new ParametricEqualizerProcessor(1000.0d, 12.0d, 0.5d), offCenter);
        double narrow = rmsRatioAfter(new ParametricEqualizerProcessor(1000.0d, 12.0d, 8.0d), offCenter);
        assertTrue("a wider band (low Q) should affect the off-center tone more: wide=" + wide
                + " narrow=" + narrow, wide > narrow);
    }

    @Test
    public void freshProcessorsProduceReproducibleResults() {
        short[] input = sine(1000.0d, 4000);
        short[] first = process(new ParametricEqualizerProcessor(1000.0d, 9.0d, 1.5d), input);
        short[] second = process(new ParametricEqualizerProcessor(1000.0d, 9.0d, 1.5d), input);
        assertArrayEquals("a fresh processor resets state and reproduces the result", first, second);
    }

    // ------------------------------------------------------------------ shelves

    @Test
    public void lowShelfAffectsLowsMoreThanHighs() {
        double low = rmsRatioAfter(new LowShelfEqualizerProcessor(500.0d, 12.0d, 1.0d), sine(150.0d, 6000));
        double high = rmsRatioAfter(new LowShelfEqualizerProcessor(500.0d, 12.0d, 1.0d), sine(6000.0d, 6000));
        assertTrue("low-shelf should boost lows more than highs: low=" + low + " high=" + high, low > high);
        assertTrue("lows are boosted", low > 1.5d);
    }

    @Test
    public void highShelfAffectsHighsMoreThanLows() {
        double high = rmsRatioAfter(new HighShelfEqualizerProcessor(3000.0d, 12.0d, 1.0d), sine(6500.0d, 6000));
        double low = rmsRatioAfter(new HighShelfEqualizerProcessor(3000.0d, 12.0d, 1.0d), sine(150.0d, 6000));
        assertTrue("high-shelf should boost highs more than lows: high=" + high + " low=" + low, high > low);
        assertTrue("highs are boosted", high > 1.5d);
    }

    @Test
    public void shelfZeroDbIsTransparent() {
        short[] input = sine(800.0d, 8000);
        short[] output = process(new LowShelfEqualizerProcessor(800.0d, 0.0d, 1.0d), input);
        int maxDiff = 0;
        for (int i = 0; i < input.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(input[i] - output[i]));
        }
        assertTrue("0 dB shelf must be transparent, max diff " + maxDiff, maxDiff <= 2);
    }

    // ------------------------------------------------------------------ stability

    @Test
    public void impulseResponseStaysFiniteAndDecays() {
        short[] impulse = new short[N];
        impulse[0] = 30000;
        short[] output = process(new ParametricEqualizerProcessor(1000.0d, 18.0d, 6.0d), impulse);
        for (short sample : output) {
            assertTrue("samples remain within PCM range", sample >= Short.MIN_VALUE && sample <= Short.MAX_VALUE);
        }
        // Tail must have died away well before the end.
        for (int i = N - 200; i < N; i++) {
            assertTrue("impulse tail decays", Math.abs(output[i]) < 50);
        }
    }

    @Test
    public void longSilenceDoesNotDrift() {
        short[] silence = new short[N];
        short[] output = process(new HighShelfEqualizerProcessor(4000.0d, 12.0d, 1.0d), silence);
        for (short sample : output) {
            assertEquals(0, sample);
        }
    }

    @Test
    public void frequencyAtOrAboveNyquistBypassesInsteadOfCrashing() {
        short[] input = sine(2000.0d, 4000);
        short[] output = process(new ParametricEqualizerProcessor(9000.0d, 12.0d, 1.0d), input); // 9 kHz > 8 kHz Nyquist
        assertArrayEquals("an out-of-range band must pass audio through unchanged", input, output);
    }

    @Test
    public void invalidParametersThrowFromTheCoefficientFactory() {
        assertRejected(new Runnable() {
            public void run() {
                BiquadCoefficients.peaking(RATE, 1000.0d, 6.0d, 0.0d); // Q = 0
            }
        });
        assertRejected(new Runnable() {
            public void run() {
                BiquadCoefficients.lowShelf(RATE, 1000.0d, 6.0d, 0.0d); // slope = 0
            }
        });
        assertRejected(new Runnable() {
            public void run() {
                BiquadCoefficients.peaking(RATE, RATE, 6.0d, 1.0d); // frequency == rate (> Nyquist)
            }
        });
    }

    @Test
    public void validCoefficientsAreFinite() {
        BiquadCoefficients c = BiquadCoefficients.highShelf(48000, 6000.0d, -6.0d, 1.0d);
        assertTrue(isFinite(c.b0) && isFinite(c.b1) && isFinite(c.b2) && isFinite(c.a1) && isFinite(c.a2));
    }

    // ------------------------------------------------------------------ helpers

    private static double rmsRatioAfter(Pcm16Processor processor, short[] input) {
        short[] output = process(processor, input);
        double before = rmsSecondHalf(input);
        double after = rmsSecondHalf(output);
        return before == 0.0d ? 0.0d : after / before;
    }

    private static short[] process(Pcm16Processor processor, short[] input) {
        short[] copy = input.clone();
        processor.process(copy, copy.length, MONO);
        return copy;
    }

    private static short[] sine(double frequencyHz, int length) {
        short[] samples = new short[length];
        for (int i = 0; i < length; i++) {
            samples[i] = (short) Math.round(8000.0d * Math.sin(2.0d * Math.PI * frequencyHz * i / RATE));
        }
        return samples;
    }

    private static double rmsSecondHalf(short[] samples) {
        long sum = 0;
        int from = samples.length / 2; // skip the filter transient
        int count = 0;
        for (int i = from; i < samples.length; i++) {
            sum += (long) samples[i] * samples[i];
            count++;
        }
        return count == 0 ? 0.0d : Math.sqrt((double) sum / count);
    }

    private static boolean isFinite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    private static void assertRejected(Runnable action) {
        try {
            action.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // ok
        }
    }
}
