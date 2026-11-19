package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/** Offline WPE dereverberation reduces the late reverberant tail of a synthetically reverberated signal. */
public class WpeDereverberationTest {

    private static final int RATE = 16000;
    private static final PcmAudioFormat MONO = new PcmAudioFormat(RATE, 1, 16);

    @Test
    public void spectrogramReconstructsTheInput() {
        double[] mono = new double[8000];
        int state = 4242;
        for (int i = 0; i < mono.length; i++) {
            state = state * 1103515245 + 12345;
            mono[i] = ((state >> 16) % 4000) - 2000;
        }
        ShortTimeSpectrogram spec = ShortTimeSpectrogram.forward(mono, 512, 128);
        double[] out = spec.inverse();
        double err = 0.0d;
        double energy = 0.0d;
        for (int i = 0; i < mono.length; i++) {
            err += (out[i] - mono[i]) * (out[i] - mono[i]);
            energy += mono[i] * mono[i];
        }
        assertTrue("STFT reconstruction error small: " + (err / energy), err / energy < 1.0e-6d);
    }

    @Test
    public void reducesTheLateReverberantTail() {
        short[] reverberant = reverberate(dryBursts(), 0.06d);
        double before = tailToOnset(reverberant);

        short[] work = reverberant.clone();
        WpeDereverberationSettings settings = new WpeDereverberationSettings(
                WpeDereverberationSettings.Mode.OFFLINE, 1.0d, 2, 10, 4, 0.0d, false, 0.05d, 0.3d, 64);
        new WpeDereverberation(settings).process(work, work.length, MONO, SpeechGate.NEVER);
        double after = tailToOnset(work);

        short[] recon = reverberant.clone();
        WpeDereverberationSettings zero = new WpeDereverberationSettings(
                WpeDereverberationSettings.Mode.OFFLINE, 0.0d, 2, 10, 4, 0.0d, false, 0.05d, 0.3d, 64);
        new WpeDereverberation(zero).process(recon, recon.length, MONO, SpeechGate.NEVER);
        int wpeVsRecon = 0;
        for (int i = 0; i < work.length; i++) {
            if (work[i] != recon[i]) {
                wpeVsRecon++;
            }
        }
        assertTrue("WPE core changed the signal vs pure reconstruction (diff=" + wpeVsRecon + ")",
                wpeVsRecon > 1000);
        assertTrue("late tail reduced: before=" + before + " after=" + after, after < before * 0.85d);
    }

    @Test
    public void blockAdaptiveModeAlsoReducesTheTailAndDiffersFromOffline() {
        short[] reverberant = reverberate(dryBursts(), 0.06d);
        double before = tailToOnset(reverberant);

        short[] blockAdaptive = reverberant.clone();
        new WpeDereverberation(new WpeDereverberationSettings(
                WpeDereverberationSettings.Mode.BLOCK_ADAPTIVE, 1.0d, 2, 10, 4, 0.0d, false, 0.05d, 0.3d, 48))
                .process(blockAdaptive, blockAdaptive.length, MONO, SpeechGate.NEVER);
        double after = tailToOnset(blockAdaptive);
        assertTrue("block-adaptive reduces the tail: before=" + before + " after=" + after,
                after < before * 0.9d);

        short[] offline = reverberant.clone();
        new WpeDereverberation(new WpeDereverberationSettings(
                WpeDereverberationSettings.Mode.OFFLINE, 1.0d, 2, 10, 4, 0.0d, false, 0.05d, 0.3d, 48))
                .process(offline, offline.length, MONO, SpeechGate.NEVER);
        int diff = 0;
        for (int i = 0; i < offline.length; i++) {
            if (offline[i] != blockAdaptive[i]) {
                diff++;
            }
        }
        assertTrue("block-adaptive result differs from offline (diff=" + diff + ")", diff > 1000);
    }

    @Test
    public void streamingModeReducesTheTailWithBoundedHistory() {
        short[] reverberant = reverberate(dryBursts(), 0.06d);
        double before = tailToOnset(reverberant);
        short[] work = reverberant.clone();
        new WpeDereverberation(new WpeDereverberationSettings(
                WpeDereverberationSettings.Mode.STREAMING, 1.0d, 2, 10, 3, 0.0d, false, 0.05d, 0.4d, 48))
                .process(work, work.length, MONO, SpeechGate.NEVER);
        double after = tailToOnset(work);
        // Streaming uses only bounded history + a short look-ahead, so it is intentionally gentler than the
        // offline pass; it still removes a clear part of the late tail.
        assertTrue("streaming reduces the tail: before=" + before + " after=" + after,
                after < before * 0.95d);
    }

    private static short[] dryBursts() {
        short[] dry = new short[32000];
        int state = 97;
        for (int b = 0; b < 8; b++) {
            int start = b * 4000;
            for (int i = 0; i < 200; i++) {
                state = state * 1103515245 + 12345;
                dry[start + i] = (short) (((state >> 16) % 16000) - 8000);
            }
        }
        return dry;
    }

    /** Convolve with an exponential-decay impulse response, then scale to a safe amplitude. */
    private static short[] reverberate(short[] dry, double tauSeconds) {
        int irLen = 3000;
        double tau = tauSeconds * RATE;
        double[] ir = new double[irLen];
        for (int n = 0; n < irLen; n++) {
            ir[n] = Math.exp(-n / tau);
        }
        double[] wet = new double[dry.length];
        for (int i = 0; i < dry.length; i++) {
            if (dry[i] == 0) {
                continue;
            }
            double v = dry[i];
            int max = Math.min(irLen, dry.length - i);
            for (int n = 0; n < max; n++) {
                wet[i + n] += v * ir[n];
            }
        }
        double peak = 0.0d;
        for (double w : wet) {
            peak = Math.max(peak, Math.abs(w));
        }
        double scale = peak > 0.0d ? 8000.0d / peak : 1.0d;
        short[] out = new short[dry.length];
        for (int i = 0; i < dry.length; i++) {
            out[i] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, Math.round(wet[i] * scale)));
        }
        return out;
    }

    /** Ratio of energy in the decay region after each burst to the energy at the burst onset. */
    private static double tailToOnset(short[] s) {
        double tail = 0.0d;
        double onset = 0.0d;
        for (int b = 0; b < 8; b++) {
            int start = b * 4000;
            for (int i = 0; i < 300 && start + i < s.length; i++) {
                onset += (double) s[start + i] * s[start + i];
            }
            for (int i = 500; i < 3800 && start + i < s.length; i++) {
                tail += (double) s[start + i] * s[start + i];
            }
        }
        return tail / Math.max(1.0d, onset);
    }
}
