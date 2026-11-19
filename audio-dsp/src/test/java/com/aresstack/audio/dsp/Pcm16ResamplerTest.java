package com.aresstack.audio.dsp;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Linear resampling: correct output length for integer and non-integer ratios, tail preserved. */
public class Pcm16ResamplerTest {

    @Test
    public void resamples48kTo16kWithOneThirdLength() {
        short[] input = ramp(300);
        short[] output = Pcm16Resampler.resample(input, 48000, 16000);
        assertEquals(100, output.length);          // 300 * 16000 / 48000
        assertEquals(0, output[0]);
        // The tail is kept (not truncated): the last output is near the last input value.
        assertTrue("tail=" + output[output.length - 1], output[output.length - 1] > 260);
    }

    @Test
    public void resamples44100To16000WithNonIntegerRatio() {
        short[] input = ramp(441);
        short[] output = Pcm16Resampler.resample(input, 44100, 16000);
        assertEquals(160, output.length);          // round(441 * 16000 / 44100)
        assertEquals(0, output[0]);
        assertTrue(output[output.length - 1] > 400);
    }

    @Test
    public void equalRatesCopyThrough() {
        short[] input = ramp(50);
        short[] output = Pcm16Resampler.resample(input, 16000, 16000);
        assertEquals(50, output.length);
        assertEquals(input[49], output[49]);
    }

    @Test
    public void emptyInputStaysEmpty() {
        assertEquals(0, Pcm16Resampler.resample(new short[0], 48000, 16000).length);
    }

    private static short[] ramp(int length) {
        short[] values = new short[length];
        for (int i = 0; i < length; i++) {
            values[i] = (short) i;
        }
        return values;
    }
}
