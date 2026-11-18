package com.aresstack.windirectml.inference.whisper.audio;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BluesteinFourierTransformTest {
    @Test
    void computesWhisperFourHundredPointTransformAgainstDirectDft() {
        int n = 400;
        BluesteinFourierTransform fft = new BluesteinFourierTransform(n);
        double[] real = new double[n];
        double[] imag = new double[n];
        for (int i = 0; i < n; i++) {
            real[i] = Math.sin(0.071 * i) + 0.31 * Math.cos(0.37 * i);
            imag[i] = 0.12 * Math.sin(0.019 * i);
        }
        double[] originalReal = real.clone();
        double[] originalImag = imag.clone();
        fft.forward(real, imag);
        for (int k = 0; k < n; k += 7) {
            double expectedReal = 0;
            double expectedImag = 0;
            for (int i = 0; i < n; i++) {
                double angle = -2.0 * Math.PI * k * i / n;
                expectedReal += originalReal[i] * Math.cos(angle) - originalImag[i] * Math.sin(angle);
                expectedImag += originalReal[i] * Math.sin(angle) + originalImag[i] * Math.cos(angle);
            }
            assertEquals(expectedReal, real[k], 1.0e-8);
            assertEquals(expectedImag, imag[k], 1.0e-8);
        }
        fft.inverse(real, imag);
        for (int i = 0; i < n; i++) {
            assertEquals(originalReal[i], real[i], 1.0e-9);
            assertEquals(originalImag[i], imag[i], 1.0e-9);
        }
    }

    @Test
    void rejectsIncorrectComplexBuffers() {
        BluesteinFourierTransform fft = new BluesteinFourierTransform(400);
        assertThrows(IllegalArgumentException.class, () -> fft.forward(new double[400], new double[399]));
    }
}
