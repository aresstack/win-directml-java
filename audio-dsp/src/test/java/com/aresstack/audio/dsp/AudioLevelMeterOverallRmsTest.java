package com.aresstack.audio.dsp;

import com.aresstack.audio.domain.PcmAudioFormat;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/** The overall RMS accumulates across every processed block, not just the last frame. */
public class AudioLevelMeterOverallRmsTest {

    private final PcmAudioFormat format = PcmAudioFormat.speechDefault();

    @Test
    public void overallRmsSpansAllBlocks() {
        AudioLevelMeter meter = new AudioLevelMeter();
        meter.process(constant(1000, 100), 100, format);   // loud block
        meter.process(constant(0, 100), 100, format);       // silent block
        // RMS of [1000×100, 0×100] = sqrt((100*1000^2)/200) = 1000/sqrt(2) ≈ 707.
        assertEquals(707.1d, meter.getOverallRms(), 1.0d);
        // The last-frame RMS only reflects the silent block.
        assertEquals(0.0d, meter.getLastFrameRms(), 0.001d);
    }

    @Test
    public void resetClearsOverallRms() {
        AudioLevelMeter meter = new AudioLevelMeter();
        meter.process(constant(500, 50), 50, format);
        meter.reset();
        assertEquals(0.0d, meter.getOverallRms(), 0.001d);
    }

    private static short[] constant(int value, int length) {
        short[] samples = new short[length];
        for (int i = 0; i < length; i++) {
            samples[i] = (short) value;
        }
        return samples;
    }
}
