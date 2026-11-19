package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public final class ButterworthFilterProcessorTest {

    @Test
    public void lowPassAboveNyquistBecomesNoOpForLowRateInput() {
        short[] samples = new short[]{100, -200, 300};
        short[] original = samples.clone();

        ButterworthFilterProcessor.lowPass(4, 7200.0d)
                .process(samples, samples.length, new PcmAudioFormat(8000, 1, 16));

        assertArrayEquals(original, samples);
    }
}
