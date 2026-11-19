package com.aresstack.audio.dsp;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** The anti-aliasing low-pass removes content above the destination Nyquist and keeps the pass-band. */
public class Pcm16LowPassFilterTest {

    private static short[] sine(int rate, double freq, int samples, double amplitude) {
        short[] out = new short[samples];
        for (int i = 0; i < samples; i++) {
            out[i] = (short) Math.round(amplitude * Math.sin(2.0 * Math.PI * freq * i / rate));
        }
        return out;
    }

    private static double rms(short[] samples) {
        double sum = 0.0;
        for (int i = 0; i < samples.length; i++) {
            sum += (double) samples[i] * samples[i];
        }
        return Math.sqrt(sum / Math.max(1, samples.length));
    }

    @Test
    public void attenuatesContentAboveCutoff() {
        // 12 kHz at 48 kHz is above the 16 kHz destination Nyquist (8 kHz): it must be strongly reduced.
        short[] high = sine(48000, 12000, 48000, 10000);
        double before = rms(high);
        double after = rms(Pcm16LowPassFilter.filter(high, 48000, 0.45 * 16000));
        assertTrue("high tone should be heavily attenuated (before=" + before + ", after=" + after + ")",
                after < before * 0.15);
    }

    @Test
    public void preservesPassBand() {
        // 1 kHz is well inside the pass-band: its level should be roughly preserved.
        short[] low = sine(48000, 1000, 48000, 10000);
        double before = rms(low);
        double after = rms(Pcm16LowPassFilter.filter(low, 48000, 0.45 * 16000));
        assertTrue("pass-band level preserved (before=" + before + ", after=" + after + ")",
                after > before * 0.85);
    }

    @Test
    public void cutoffAtOrAboveNyquistIsANoOp() {
        short[] in = sine(16000, 1000, 1600, 8000);
        short[] out = Pcm16LowPassFilter.filter(in, 16000, 9000); // above Nyquist (8 kHz)
        assertEquals(in.length, out.length);
        for (int i = 0; i < in.length; i++) {
            assertEquals(in[i], out[i]);
        }
    }

    @Test
    public void resamplerRemovesAliasingWhenDownsampling() {
        // Without the low-pass a 12 kHz tone at 48 kHz would alias to 4 kHz at 16 kHz; with it, the 16 kHz
        // output is near-silent for that input.
        short[] high = sine(48000, 12000, 48000, 10000);
        double resampled = rms(Pcm16Resampler.resample(high, 48000, 16000));
        assertTrue("aliasing removed (rms=" + resampled + ")", resampled < 2000.0);
    }
}
