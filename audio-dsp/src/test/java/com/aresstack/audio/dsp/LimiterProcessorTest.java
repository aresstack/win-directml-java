package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LimiterProcessorTest {

    @Test
    public void clampsAboveCeilingAndKeepsBelow() {
        PcmAudioFormat format = PcmAudioFormat.speechDefault();
        LimiterProcessor processor = new LimiterProcessor(30000);

        short[] samples = {0, 1000, -1000, 29999, -29999, 30000, 32000, -32000, Short.MAX_VALUE, Short.MIN_VALUE};
        processor.process(samples, samples.length, format);

        assertEquals(0, samples[0]);
        assertEquals(1000, samples[1]);
        assertEquals(-1000, samples[2]);
        assertEquals(29999, samples[3]);
        assertEquals(-29999, samples[4]);
        assertEquals(30000, samples[5]);
        assertEquals(30000, samples[6]);
        assertEquals(-30000, samples[7]);
        assertEquals(30000, samples[8]);
        assertEquals(-30000, samples[9]);
    }
}
