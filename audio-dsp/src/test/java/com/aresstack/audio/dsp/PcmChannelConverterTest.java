package com.aresstack.audio.dsp;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Stereo→mono down-mix averages channels; mono passes through; partial trailing frame is dropped. */
public class PcmChannelConverterTest {

    @Test
    public void averagesStereoToMono() {
        // frames: (100,200)->150, (1000,-1000)->0, (30000,30000)->30000
        short[] stereo = {100, 200, 1000, -1000, 30000, 30000};
        short[] mono = PcmChannelConverter.downmixToMono(stereo, stereo.length, 2);
        assertArrayEquals(new short[]{150, 0, 30000}, mono);
    }

    @Test
    public void monoPassesThroughUnchanged() {
        short[] mono = {5, -5, 32767, -32768};
        assertArrayEquals(mono, PcmChannelConverter.downmixToMono(mono, mono.length, 1));
    }

    @Test
    public void dropsIncompleteTrailingFrame() {
        // 5 samples, 2 channels -> 2 complete frames, last lone sample dropped.
        short[] stereo = {10, 20, 30, 40, 50};
        short[] mono = PcmChannelConverter.downmixToMono(stereo, stereo.length, 2);
        assertEquals(2, mono.length);
        assertArrayEquals(new short[]{15, 35}, mono);
    }
}
