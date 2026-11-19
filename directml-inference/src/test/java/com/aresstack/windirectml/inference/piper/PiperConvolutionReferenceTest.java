package com.aresstack.windirectml.inference.piper;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class PiperConvolutionReferenceTest {

    @Test
    void computesCrossCorrelationWithBiasAndPadding() {
        float[][] input = {{1, 2, 3, 4}};
        float[][][] weights = {{{2, 1}}};

        float[][] output = PiperConv1dReference.apply(input, weights,
                new float[]{1}, 1, 1, 1, 0, 1);

        assertArrayEquals(new float[]{2, 5, 8, 11}, output[0], 0.0001f);
    }

    @Test
    void isolatesGroupedChannels() {
        float[][] input = {{1, 2}, {10, 20}};
        float[][][] weights = {{{2}}, {{3}}};

        float[][] output = PiperConv1dReference.apply(input, weights,
                null, 1, 1, 0, 0, 2);

        assertArrayEquals(new float[]{2, 4}, output[0], 0.0001f);
        assertArrayEquals(new float[]{30, 60}, output[1], 0.0001f);
    }

    @Test
    void computesTransposedConvolutionAndDecoderUpsampling() {
        float[][] input = {{1, 2}};
        float[][][] weights = {{{1, 2}}};

        float[][] output = PiperConvTranspose1dReference.apply(input, weights,
                null, 2, 1, 0, 0, 1, 0);

        assertArrayEquals(new float[]{1, 2, 2, 4}, output[0], 0.0001f);
    }

    @Test
    void validatesInvalidShapesRatherThanReturningSilentAudio() {
        assertThrows(IllegalArgumentException.class, () -> PiperConv1dReference.apply(
                new float[][]{{1, 2}}, new float[][][]{{{1}}},
                null, 0, 1, 0, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> PiperConvTranspose1dReference.apply(
                new float[][]{{1, 2}}, new float[][][]{{{1}}},
                null, 2, 1, 0, 0, 1, 2));
    }
}
