package com.aresstack.audio.dsp;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

/** Probe the complex linear solver and the spectrogram mutation path used by WPE. */
public class WpeSolverProbeTest {

    @Test
    public void solvesASimpleComplexSystem() {
        // R = [[2+0i, 0], [0, 4+0i]], p = [2+2i, 4]; expect g = [1+1i, 1].
        double[][] rRe = {{2.0d, 0.0d}, {0.0d, 4.0d}};
        double[][] rIm = {{0.0d, 0.0d}, {0.0d, 0.0d}};
        double[] pRe = {2.0d, 4.0d};
        double[] pIm = {2.0d, 0.0d};
        double[] gRe = new double[2];
        double[] gIm = new double[2];
        boolean ok = WpeDereverberation.solveComplex(rRe, rIm, pRe, pIm, 2, gRe, gIm);
        assertTrue("solver succeeds", ok);
        assertTrue("g0 real ~1: " + gRe[0], Math.abs(gRe[0] - 1.0d) < 1.0e-9d);
        assertTrue("g0 imag ~1: " + gIm[0], Math.abs(gIm[0] - 1.0d) < 1.0e-9d);
        assertTrue("g1 real ~1: " + gRe[1], Math.abs(gRe[1] - 1.0d) < 1.0e-9d);
    }

    @Test
    public void zeroingABinChangesTheReconstruction() {
        double[] mono = new double[8000];
        for (int i = 0; i < mono.length; i++) {
            mono[i] = 6000.0d * Math.sin(2.0d * Math.PI * 400.0d * i / 16000.0d);
        }
        ShortTimeSpectrogram spec = ShortTimeSpectrogram.forward(mono, 512, 128);
        int k = (int) Math.round(400.0d / (16000.0d / 512)); // the 400 Hz bin
        for (int t = 0; t < spec.getFrameCount(); t++) {
            spec.realFrame(t)[k] = 0.0d;
            spec.imagFrame(t)[k] = 0.0d;
            int mirror = 512 - k;
            spec.realFrame(t)[mirror] = 0.0d;
            spec.imagFrame(t)[mirror] = 0.0d;
        }
        double[] out = spec.inverse();
        double energy = 0.0d;
        for (int i = 2000; i < 6000; i++) {
            energy += out[i] * out[i];
        }
        double inEnergy = 0.0d;
        for (int i = 2000; i < 6000; i++) {
            inEnergy += mono[i] * mono[i];
        }
        assertTrue("zeroing the tone bin removes most energy: " + (energy / inEnergy),
                energy / inEnergy < 0.2d);
    }
}
