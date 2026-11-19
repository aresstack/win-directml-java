package com.aresstack.audio.dsp;

/** Analysis/synthesis window functions for short-time spectral processing. */
public final class WindowFunctions {

    private WindowFunctions() {
    }

    /** @return a periodic Hann window of the given size (suitable for overlap-add STFT). */
    public static double[] hann(int size) {
        double[] window = new double[size];
        for (int i = 0; i < size; i++) {
            window[i] = 0.5d - 0.5d * Math.cos(2.0d * Math.PI * i / size);
        }
        return window;
    }
}
